# -*- coding: utf-8 -*-
"""
core/gamma.py

Apply gamma on linear RGB (counts domain 0..WL):
    y = (x / WL_in) ** gamma * WL_out

- Saving:
  * 8-bit: Pillow PNG
  * ≥10-bit: OpenCV 16-bit TIFF (works around Pillow uint16 RGB limits)
- Preview:
  * "before": source is linear → apply sRGB OETF only for visualization (preview_before_srgb.png)

Note: this file does not change the meaning of meta['white_level'] (kept same as input by default).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image


def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def _quant_max_for_bitdepth(bit: int) -> int:
    if bit not in (8, 10, 12, 14, 16):
        raise ValueError(f"Invalid bit depth: {bit}")
    return (1 << bit) - 1

def _quantize_for_saving(img: np.ndarray, wl_out: float, bit: int) -> np.ndarray:
    """Normalize by WL_out, clamp to [0,1], then quantize to the target bit depth."""
    if wl_out <= 0:
        raise ValueError("Quantization failed: WL_out must be positive")
    maxv = _quant_max_for_bitdepth(bit)
    img01 = np.clip(img.astype(np.float32) / float(wl_out), 0.0, 1.0)
    return _to_uint(img01 * float(maxv), maxv)

def _to_uint(img: np.ndarray, maxv: int) -> np.ndarray:
    """Clamp to [0,maxv], round, and cast to uint8/uint16 based on maxv."""
    x = np.rint(np.clip(img, 0.0, float(maxv)))
    return x.astype(np.uint8 if maxv <= 255 else np.uint16, copy=False)


def _save_srgb_preview_8bit(rgb_linear: np.ndarray,
                            wl_in: float,
                            out_path: Path) -> Path:
    """
    Linear RGB (0..WL_in) → sRGB OETF for human-friendly preview.
    Does not affect the high-bit-depth output.
    """
    wl = float(wl_in) if wl_in and wl_in > 0 else 1.0
    x = np.clip(rgb_linear.astype(np.float32) / wl, 0.0, 1.0)

    # sRGB OETF
    a = 0.055
    y = np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)

    img8 = (y * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(img8), mode="RGB").save(out_path)
    return out_path


def gamma_correct_core(x: np.ndarray,
                       gamma: float,
                       white_level_in: float,
                       white_level_out: float,
                       clamp: bool = True,
                       neg_soft_floor: float = 0.0) -> np.ndarray:
    """
    Core gamma: normalize by WL_in → power curve → rescale to WL_out.

    Extra:
    - To avoid hard zeroing tiny negatives (which may cause color bias),
      set neg_soft_floor>0 (e.g., 1e-6) to lift [-neg_soft_floor, 0) before power.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if white_level_in <= 0 or white_level_out <= 0:
        raise ValueError("white_level must be positive")

    x = x.astype(np.float32, copy=False)

    # Normalize (cap upper to 1; keep negatives for handling below)
    x_norm = x / float(white_level_in)
    if clamp:
        x_norm = np.minimum(x_norm, 1.0)

    # Soft lift small negatives to reduce per-channel clipping bias
    if neg_soft_floor > 0.0:
        neg_mask_small = (x_norm < 0.0) & (x_norm >= -float(neg_soft_floor))
        if np.any(neg_mask_small):
            x_norm = x_norm.copy()
            x_norm[neg_mask_small] = float(neg_soft_floor)

    # Power curve on positive values only; non-positive become 0 if clamp=True
    y_norm = np.zeros_like(x_norm, dtype=np.float32)
    pos = x_norm > 0.0
    if np.any(pos):
        y_norm[pos] = np.power(x_norm[pos], float(gamma))

    if clamp:
        y_norm = np.clip(y_norm, 0.0, 1.0)

    # Rescale back to counts
    y = y_norm * float(white_level_out)
    return y.astype(np.float32, copy=False)


def run_gamma_rgb(rgb_in: np.ndarray,
                  meta_in: Dict[str, Any],
                  out_dir: str,
                  *,
                  gamma: float = 0.7,
                  save_bitdepth: Optional[int] = 8,
                  clamp: bool = True,
                  neg_soft_floor: float = 0.0,
                  save_filename: str = "rgb_gamma"
                  ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    ISP entry: apply gamma to linear RGB (counts 0..WL) after DEMOSAIC/CCM.

    Args:
      - rgb_in: HxWx3 float/int in [0..meta['white_level']]
      - meta_in: must include 'white_level'
      - out_dir: output folder
      - gamma: power exponent (>0)
      - save_bitdepth: 8/10/12/14/16 to save; None to skip saving
      - clamp: clamp to [0,1] during normalize/power
      - neg_soft_floor: small negative lift near 0 to reduce color bias
      - save_filename: basename for saved files

    Returns: (rgb_out_float32, meta_out, info)
    """
    if rgb_in is None or meta_in is None:
        raise ValueError("run_gamma_rgb: empty input")
    if rgb_in.ndim != 3 or rgb_in.shape[2] != 3:
        raise ValueError(f"run_gamma_rgb: expect HxWx3, got {rgb_in.shape}")

    WL_in = float(meta_in.get("white_level", 0))
    WL_out = float(meta_in.get("white_level", 0))
    if WL_in <= 0:
        raise ValueError("run_gamma_rgb: meta['white_level'] missing or invalid")

    # Keep WL_out semantics aligned with input (original behavior).
    # If you prefer WL_out based on save_bitdepth, switch to the commented logic below.
    # if save_bitdepth in (8, 10, 12, 14, 16):
    #     WL_out = float(_quant_max_for_bitdepth(int(save_bitdepth)))
    # else:
    #     WL_out = WL_in
    #     save_bitdepth = None

    # Core processing
    y = gamma_correct_core(
        rgb_in, gamma=gamma,
        white_level_in=WL_in,
        white_level_out=WL_out,
        clamp=clamp,
        neg_soft_floor=neg_soft_floor
    )

    # Meta
    meta_out = dict(meta_in)
    meta_out["white_level"] = WL_out
    meta_out["gamma"] = float(gamma)
    meta_out["is_gamma_encoded"] = True

    info: Dict[str, Any] = {
        "gamma": float(gamma),
        "clamp": bool(clamp),
        "neg_soft_floor": float(neg_soft_floor),
        "white_level_in": WL_in,
        "white_level_out": WL_out,
        "saved": False,
        "saved_path": None,
        "saved_bitdepth": None,
        "preview_before_srgb_path": None,
    }

    # Optional saving
    if save_bitdepth is not None:
        _ensure_dir(out_dir)
        out_dir_p = Path(out_dir)

        if save_bitdepth == 8:
            # 8-bit PNG via Pillow
            q = _quantize_for_saving(y, WL_out, 8).astype(np.uint8, copy=False)
            out_path = out_dir_p / f"{save_filename}_8bit.png"
            Image.fromarray(np.ascontiguousarray(q), mode="RGB").save(out_path)
        else:
            # ≥10-bit: write as 16-bit TIFF via OpenCV (container), values are quantized
            import cv2  # lazy import
            maxv = _quant_max_for_bitdepth(int(save_bitdepth))
            q16 = _to_uint(y, maxv).astype(np.uint16, copy=False)
            # OpenCV writes BGR; channel order swap is harmless to data content
            q16_bgr = q16[..., ::-1]
            out_path = out_dir_p / f"{save_filename}_{int(save_bitdepth)}bit.tiff"
            ok = cv2.imwrite(str(out_path), q16_bgr)
            if not ok:
                raise IOError(f"Save failed: {out_path}")

        info.update({
            "saved": True,
            "saved_path": str(out_path),
            "saved_bitdepth": int(save_bitdepth),
        })

    # Preview: linear → sRGB (before)
    try:
        out_dir_p = Path(out_dir)
        out_before = out_dir_p / "preview_before_srgb.png"
        _ensure_dir(out_dir_p)
        _save_srgb_preview_8bit(rgb_in.astype(np.float32), WL_in, out_before)
        info["preview_before_srgb_path"] = str(out_before)
    except Exception:
        pass

    return y, meta_out, info
