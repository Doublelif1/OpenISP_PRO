# -*- coding: utf-8 -*-
"""
core.contrast
— Contrast adjustment in Y'CbCr 4:4:4 (planar, uint8).
Placement: after SAT, operate in YUV and modify Y only; pass U/V through to avoid chroma shifts.
Methods:
  - linear:   out = (in - midpoint) * factor + midpoint
  - sigmoid:  out = 1 / (1 + exp(gain * (cutoff - in)))
  - gamma:    out = in ** gamma
  - hist_eq:  global histogram equalization on Y
API returns ((Y,U,V), meta_out, info). Optionally saves PNG / planar .yuv / RGB previews.
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("core.contrast")


# -------------------- I/O & preview helpers --------------------
def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _save_planar_yuv444(path: str, Y: np.ndarray, U: np.ndarray, V: np.ndarray):
    Yc, Uc, Vc = np.ascontiguousarray(Y), np.ascontiguousarray(U), np.ascontiguousarray(V)
    with open(path, "wb") as f:
        f.write(Yc.tobytes()); f.write(Uc.tobytes()); f.write(Vc.tobytes())


def _yuv444_to_rgb_u8(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                      standard: str = "bt601",
                      range_mode: str = "full") -> np.ndarray:
    """Lightweight YUV444→RGB' 8-bit for preview only."""
    Yf = np.asarray(Y, np.float32)
    Cb = np.asarray(U, np.float32)
    Cr = np.asarray(V, np.float32)

    m = (range_mode or "full").lower()
    if m == "limited":
        Yf  = (Yf - 16.0)  * (255.0 / 219.0)
        Cbf = (Cb - 128.0) * (255.0 / 224.0)
        Crf = (Cr - 128.0) * (255.0 / 224.0)
        Cb  = Cbf + 128.0
        Cr  = Crf + 128.0
    elif m != "full":
        logger.warning("[contrast] unknown range_mode '%s', assume full", range_mode)

    std = (standard or "bt601").lower()
    KRKGKB = {
        "bt601":  (0.299000, 0.587000, 0.114000),
        "bt709":  (0.212600, 0.715200, 0.072200),
        "bt2020": (0.262700, 0.678000, 0.059300),
    }
    if std not in KRKGKB:
        logger.warning("[contrast] unknown standard '%s', fallback to bt601", standard)
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
    """Resize keeping aspect ratio so max side ≤ max_side (for previews)."""
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
                          out_path: Path, standard: str, range_mode: str,
                          max_side: int = 1600) -> Path:
    """Save an RGB preview converted from YUV444."""
    rgb = _yuv444_to_rgb_u8(Y, U, V, standard=standard, range_mode=range_mode)
    rgb = _resize_max_side_uint8(rgb, max_side=max_side)
    Image.fromarray(rgb, mode="RGB").save(out_path)
    return out_path


# -------------------- Contrast core on Y channel --------------------
def _y_to_norm01(Y: np.ndarray, range_mode: str) -> np.ndarray:
    """Map Y (uint8) to full-range normalized 0..1 float."""
    Yf = Y.astype(np.float32)
    m = (range_mode or "full").lower()
    if m == "limited":
        Yf = (Yf - 16.0) * (255.0 / 219.0)  # to full-equivalent 0..255
    elif m != "full":
        logger.warning("[contrast] unknown range_mode '%s', assume full", range_mode)
    return np.clip(Yf / 255.0, 0.0, 1.0)


def _norm01_to_Y(norm01: np.ndarray, range_mode: str) -> np.ndarray:
    """Map full-equivalent 0..1 back to Y (uint8) respecting range_mode."""
    y255 = np.clip(norm01, 0.0, 1.0) * 255.0
    m = (range_mode or "full").lower()
    if m == "limited":
        Y = 16.0 + (219.0 / 255.0) * y255
    else:
        Y = y255
    return (np.clip(Y, 0.0, 255.0) + 0.5).astype(np.uint8)


def _contrast_linear(norm: np.ndarray, factor: float, midpoint: float) -> np.ndarray:
    if factor <= 0:
        raise ValueError("factor must be positive")
    return (norm - midpoint) * factor + midpoint


def _contrast_sigmoid(norm: np.ndarray, gain: float, cutoff: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(gain * (cutoff - norm)))


def _contrast_gamma(norm: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return np.power(np.clip(norm, 0.0, 1.0), gamma)


def _equalize_u8(arr_u8: np.ndarray) -> np.ndarray:
    """Global histogram equalization on uint8 array."""
    if arr_u8.dtype != np.uint8:
        raise ValueError("arr_u8 must be uint8")
    hist = np.bincount(arr_u8.ravel(), minlength=256)
    cdf = hist.cumsum()
    if cdf[-1] == 0:
        return arr_u8.copy()
    cdf_nonzero = cdf[cdf > 0]
    cdf_min = int(cdf_nonzero[0]) if cdf_nonzero.size > 0 else 0
    lut = (cdf - cdf_min) * 255.0 / max(1, (arr_u8.size - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[arr_u8]


def _apply_contrast_on_Y(Y: np.ndarray, range_mode: str,
                         method: str,
                         factor: float = 1.2,
                         midpoint: float = 0.5,
                         gain: float = 10.0,
                         cutoff: float = 0.5,
                         gamma: float = 0.9) -> np.ndarray:
    """
    Apply contrast on Y only; U/V are untouched.
    """
    method = method.lower()
    if method == "hist_eq":
        # Do HE in full-equivalent 0..255 integer domain, then map back.
        y_full_u8 = (_norm01_to_Y(_y_to_norm01(Y, range_mode), "full")).astype(np.uint8)
        y_eq_u8 = _equalize_u8(y_full_u8)
        Y_out = _norm01_to_Y(y_eq_u8.astype(np.float32) / 255.0, range_mode)
        return Y_out

    # Other methods operate in full-equivalent 0..1 float domain.
    yn = _y_to_norm01(Y, range_mode)
    if method == "linear":
        yn2 = _contrast_linear(yn, factor=factor, midpoint=midpoint)
    elif method == "sigmoid":
        yn2 = _contrast_sigmoid(yn, gain=gain, cutoff=cutoff)
    elif method == "gamma":
        yn2 = _contrast_gamma(yn, gamma=gamma)
    else:
        raise ValueError("method must be one of: linear/sigmoid/gamma/hist_eq")
    return _norm01_to_Y(yn2, range_mode)


# -------------------- Public API --------------------
def run_contrast_after_sat(
    Y_in: np.ndarray,
    U_in: np.ndarray,
    V_in: np.ndarray,
    meta_in: Dict[str, Any],
    out_dir: str,
    method: str = "linear",
    factor: float = 1.2,
    midpoint: float = 0.5,
    gain: float = 10.0,
    cutoff: float = 0.5,
    gamma: float = 0.9,
    range_mode: Optional[str] = None,
    standard: Optional[str] = None,
    save_raw: bool = True,
    save_png: bool = True,
    save_preview_rgb: bool = True,
    filename_stem: str = "yuv444_ctr",
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """
    Contrast on YUV444 after SAT (modify Y only). Matches other core.* API style.

    Returns:
        (Y_out, U_out, V_out), meta_out, info
    """
    _ensure_dir(out_dir)

    if not (isinstance(Y_in, np.ndarray) and isinstance(U_in, np.ndarray) and isinstance(V_in, np.ndarray)):
        raise TypeError("run_contrast_after_sat: Y/U/V must be numpy.ndarray")
    if not (Y_in.shape == U_in.shape == V_in.shape):
        raise ValueError(f"run_contrast_after_sat: shape mismatch: {Y_in.shape} / {U_in.shape} / {V_in.shape}")
    if Y_in.dtype != np.uint8 or U_in.dtype != np.uint8 or V_in.dtype != np.uint8:
        logger.warning("[contrast] input is not uint8; interpreting as uint8 (caller must ensure range)")

    meta_out = dict(meta_in or {})
    std  = (standard  or meta_out.get("csc_standard") or "bt601")
    rng  = (range_mode or meta_out.get("csc_range")    or meta_out.get("range") or "full")
    method = (method or "linear").lower()

    # Preview: before
    info: Dict[str, Any] = {"method": method, "range": rng, "standard": std, "saved": {},
                            "params": {"factor": factor, "midpoint": midpoint, "gain": gain,
                                       "cutoff": cutoff, "gamma": gamma}}
    if save_preview_rgb:
        try:
            prev_before = Path(out_dir) / "preview_before_ctr_yuv2rgb.png"
            _save_yuv_preview_rgb(Y_in, U_in, V_in, prev_before, std, rng)
            info["saved"]["preview_before"] = str(prev_before)
        except Exception as e:
            logger.warning("[contrast] preview(before) save failed: %s", e)

    # Process
    Y_out = _apply_contrast_on_Y(
        Y_in, rng, method,
        factor=factor, midpoint=midpoint, gain=gain, cutoff=cutoff, gamma=gamma
    )
    U_out, V_out = U_in, V_in  # pass-through

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
            logger.warning("[contrast] Y/U/V PNG save failed: %s", e)

    # Save planar .yuv
    if save_raw:
        try:
            yuv_path = Path(out_dir) / f"{filename_stem}.yuv"
            _save_planar_yuv444(str(yuv_path), Y_out, U_out, V_out)
            info["saved"]["raw"] = str(yuv_path)
        except Exception as e:
            logger.warning("[contrast] planar .yuv save failed: %s", e)

    # Preview: after
    if save_preview_rgb:
        try:
            prev_after = Path(out_dir) / "preview_after_ctr_yuv2rgb.png"
            _save_yuv_preview_rgb(Y_out, U_out, V_out, prev_after, std, rng)
            info["saved"]["preview_after"] = str(prev_after)
        except Exception as e:
            logger.warning("[contrast] preview(after) save failed: %s", e)

    # Stats
    try:
        yn0 = _y_to_norm01(Y_in, rng); yn1 = _y_to_norm01(Y_out, rng)
        info["delta_y"] = {
            "mean_abs": float(np.mean(np.abs(yn1 - yn0))),
            "max_abs": float(np.max(np.abs(yn1 - yn0))),
            "changed_ratio_ge1": float((np.abs(Y_out.astype(np.int16) - Y_in.astype(np.int16)) >= 1).mean()),
        }
    except Exception:
        pass

    # Update meta
    meta_out["contrast_method"] = method
    meta_out["contrast_params"] = dict(info["params"])

    logger.info("[CONTRAST] done | method=%s | range=%s | params=%s | saved=%s",
                method, rng, info["params"],
                {k: ('...' if isinstance(v, dict) else v) for k, v in info.get("saved", {}).items()})

    return (Y_out, U_out, V_out), meta_out, info
