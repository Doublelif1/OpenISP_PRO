# -*- coding: utf-8 -*-
"""
core.sat
Saturation adjustment in Y'CbCr 4:4:4 (planar, uint8).
- Scales U/V around 128; Y is preserved.
- Returns ((Y,U,V), meta_out, info) and can optionally save PNG/.yuv/RGB previews.
"""

from typing import Tuple, Dict, Any, Optional
import os
from pathlib import Path
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("core.sat")


# -------------------- Utilities --------------------
def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _save_planar_yuv444(path: str, Y: np.ndarray, U: np.ndarray, V: np.ndarray):
    """Write planar 4:4:4 raw as Y, then U(Cb), then V(Cr)."""
    Yc, Uc, Vc = np.ascontiguousarray(Y), np.ascontiguousarray(U), np.ascontiguousarray(V)
    with open(path, "wb") as f:
        f.write(Yc.tobytes()); f.write(Uc.tobytes()); f.write(Vc.tobytes())


def _yuv444_to_rgb_u8(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                      standard: str = "bt601",
                      range_mode: str = "full") -> np.ndarray:
    """
    Lightweight YUV444 → RGB' uint8 for preview.
    Honors bt601/bt709/bt2020 and full/limited ranges.
    """
    Yf = np.asarray(Y, np.float32)
    Cb = np.asarray(U, np.float32)
    Cr = np.asarray(V, np.float32)

    m = (range_mode or "full").lower()
    if m == "limited":
        # Map limited to full-equivalent domain for math
        Yf  = (Yf - 16.0)  * (255.0 / 219.0)
        Cbf = (Cb - 128.0) * (255.0 / 224.0)
        Crf = (Cr - 128.0) * (255.0 / 224.0)
        Cb  = Cbf + 128.0
        Cr  = Crf + 128.0
    elif m != "full":
        logger.warning("[sat] unknown range_mode '%s', assume full", range_mode)

    std = (standard or "bt601").lower()
    KRKGKB = {
        "bt601":  (0.299000, 0.587000, 0.114000),
        "bt709":  (0.212600, 0.715200, 0.072200),
        "bt2020": (0.262700, 0.678000, 0.059300),
    }
    if std not in KRKGKB:
        logger.warning("[sat] unknown standard '%s', fallback to bt601", standard)
        std = "bt601"
    Kr, Kg, Kb = KRKGKB[std]

    Cb_ = Cb - 128.0
    Cr_ = Cr - 128.0

    coef_RV = 2.0 * (1.0 - Kr)
    coef_BU = 2.0 * (1.0 - Kb)
    coef_GU = 2.0 * Kb * (1.0 - Kb) / Kg
    coef_GV = 2.0 * Kr * (1.0 - Kr) / Kg

    R = Yf + coef_RV * Cr_
    G = Yf - coef_GU * Cb_ - coef_GV * Cr_
    B = Yf + coef_BU * Cb_

    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0)
    return (rgb + 0.5).astype(np.uint8)


def _resize_max_side_uint8(img8: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Resize keeping aspect ratio; no-op if already within bound."""
    if max_side is None or max_side <= 0:
        return img8
    h, w = img8.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img8
    scale = max_side / float(m)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    _resampling = getattr(Image, "Resampling", Image)
    pil = Image.fromarray(img8).resize((new_w, new_h), resample=_resampling.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def _save_yuv_preview_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                          out_path: Path,
                          standard: str, range_mode: str, max_side: int = 1600) -> Path:
    """Convert YUV444 to RGB preview and save as PNG."""
    rgb = _yuv444_to_rgb_u8(Y, U, V, standard=standard, range_mode=range_mode)
    rgb = _resize_max_side_uint8(rgb, max_side=max_side)
    Image.fromarray(rgb, mode="RGB").save(out_path)
    return out_path


# -------------------- Core: saturation (scale U/V only) --------------------
def _sat_scale_uv(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                  s: float, range_mode: str = "full") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale chroma (U/V) around 128 by factor s; Y is unchanged.
    Input/Output are uint8 planar.
    """
    Yf  = Y.astype(np.float32)
    Uf  = U.astype(np.float32)
    Vf  = V.astype(np.float32)

    m = (range_mode or "full").lower()
    if m == "limited":
        U_full = (Uf - 128.0) * (255.0/224.0)
        V_full = (Vf - 128.0) * (255.0/224.0)
        U_adj  = 128.0 + (s * U_full) * (224.0/255.0)
        V_adj  = 128.0 + (s * V_full) * (224.0/255.0)
        Y_out  = np.clip(Yf, 16.0, 235.0)
        U_out  = np.clip(U_adj, 16.0, 240.0)
        V_out  = np.clip(V_adj, 16.0, 240.0)
    else:
        U_adj  = 128.0 + s * (Uf - 128.0)
        V_adj  = 128.0 + s * (Vf - 128.0)
        Y_out  = np.clip(Yf,  0.0, 255.0)
        U_out  = np.clip(U_adj, 0.0, 255.0)
        V_out  = np.clip(V_adj, 0.0, 255.0)

    return Y_out.astype(np.uint8), U_out.astype(np.uint8), V_out.astype(np.uint8)


# -------------------- Public API --------------------
def run_sat_after_ee(
    Y_in: np.ndarray,
    U_in: np.ndarray,
    V_in: np.ndarray,
    meta_in: Dict[str, Any],
    out_dir: str,
    s: float = 1.8,
    range_mode: Optional[str] = None,
    standard: Optional[str] = None,
    save_raw: bool = True,
    save_png: bool = True,
    save_preview_rgb: bool = True,
    filename_stem: str = "yuv444_sat",
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """
    Saturation stage after EE in YUV444. Scales U/V; keeps Y.
    Returns ((Y,U,V), meta_out, info).
    """
    _ensure_dir(out_dir)

    # Validate & normalize meta
    if not (isinstance(Y_in, np.ndarray) and isinstance(U_in, np.ndarray) and isinstance(V_in, np.ndarray)):
        raise TypeError("run_sat_after_ee: Y/U/V must be numpy.ndarray")
    if not (Y_in.shape == U_in.shape == V_in.shape):
        raise ValueError(f"run_sat_after_ee: shape mismatch: {Y_in.shape} / {U_in.shape} / {V_in.shape}")
    if Y_in.dtype != np.uint8 or U_in.dtype != np.uint8 or V_in.dtype != np.uint8:
        logger.warning("[sat] input is not uint8; will be interpreted as uint8")

    meta_out = dict(meta_in or {})
    std  = (standard or meta_out.get("csc_standard") or "bt601")
    rng  = (range_mode or meta_out.get("csc_range") or meta_out.get("range") or "full")
    s    = float(s)

    info: Dict[str, Any] = {"s": s, "standard": std, "range": rng, "saved": {}}

    # Optional: preview before
    if save_preview_rgb:
        try:
            prev_before = Path(out_dir) / "preview_before_sat_yuv2rgb.png"
            _save_yuv_preview_rgb(Y_in, U_in, V_in, prev_before, std, rng)
            info["saved"]["preview_before"] = str(prev_before)
        except Exception as e:
            logger.warning("[sat] failed to save pre-preview: %s", e)

    # Core saturation (U/V only)
    Y_out, U_out, V_out = _sat_scale_uv(Y_in, U_in, V_in, s=s, range_mode=rng)

    # Save PNGs
    if save_png:
        try:
            Image.fromarray(Y_out).save(Path(out_dir) / "Y.png")
            Image.fromarray(U_out).save(Path(out_dir) / "U.png")
            Image.fromarray(V_out).save(Path(out_dir) / "V.png")
            info["saved"]["png"] = {
                "Y": str(Path(out_dir) / "Y.png"),
                "U": str(Path(out_dir) / "U.png"),
                "V": str(Path(out_dir) / "V.png"),
            }
        except Exception as e:
            logger.warning("[sat] failed to save Y/U/V PNG: %s", e)

    # Save planar .yuv
    if save_raw:
        try:
            yuv_path = Path(out_dir) / f"{filename_stem}.yuv"
            _save_planar_yuv444(str(yuv_path), Y_out, U_out, V_out)
            info["saved"]["raw"] = str(yuv_path)
        except Exception as e:
            logger.warning("[sat] failed to save planar .yuv: %s", e)

    # Optional: preview after
    if save_preview_rgb:
        try:
            prev_after = Path(out_dir) / "preview_after_sat_yuv2rgb.png"
            _save_yuv_preview_rgb(Y_out, U_out, V_out, prev_after, std, rng)
            info["saved"]["preview_after"] = str(prev_after)
        except Exception as e:
            logger.warning("[sat] failed to save post-preview: %s", e)

    # Quick stats for logging
    try:
        dU = U_out.astype(np.float32) - U_in.astype(np.float32)
        dV = V_out.astype(np.float32) - V_in.astype(np.float32)
        info["delta_uv"] = {
            "u_changed_ratio_ge1": float((np.abs(dU) >= 1.0).mean()),
            "v_changed_ratio_ge1": float((np.abs(dV) >= 1.0).mean()),
            "u_abs_mean": float(np.mean(np.abs(dU))),
            "v_abs_mean": float(np.mean(np.abs(dV))),
            "u_abs_max": float(np.max(np.abs(dU))),
            "v_abs_max": float(np.max(np.abs(dV))),
        }
    except Exception:
        pass

    # Write back key params to meta (keep existing CSC fields)
    meta_out = dict(meta_out)
    meta_out["sat_s"] = s

    logger.info("[SAT] done | s=%.3f | std=%s | range=%s | saved=%s",
                s, std, rng, {k: ('...' if isinstance(v, dict) else v) for k, v in info.get("saved", {}).items()})

    return (Y_out, U_out, V_out), meta_out, info
