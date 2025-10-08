# -*- coding: utf-8 -*-
"""
core.hue
Hue rotation and saturation in RGB space for an ISP stage.
- Supports global tweaks, selective hue rotation, and warm highlights.
- Input: YUV444 uint8 (planar). Output: YUV444 uint8 (planar).
"""

from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("core.hue")


# -------------------- Utilities --------------------
def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _save_planar_yuv444(path: str, Y: np.ndarray, U: np.ndarray, V: np.ndarray):
    """Write planar 4:4:4 as raw .yuv: Y, then U, then V."""
    Yc, Uc, Vc = np.ascontiguousarray(Y), np.ascontiguousarray(U), np.ascontiguousarray(V)
    with open(path, "wb") as f:
        f.write(Yc.tobytes())
        f.write(Uc.tobytes())
        f.write(Vc.tobytes())


def smoothstep(edge0, edge1, x):
    """Smooth step between two edges (Hermite interpolation)."""
    t = np.clip((x - edge0) / np.maximum(1e-6, (edge1 - edge0)), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def circ_dist_deg(a, b):
    """Shortest circular distance between hues in degrees (0..180]."""
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


# -------------------- YUV <-> RGB --------------------
def _yuv444_to_rgb_f32(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                       standard: str = "bt601", range_mode: str = "full") -> np.ndarray:
    """
    Convert YUV444 (uint8) → RGB float32 in [0,1].
    Honors bt601/bt709/bt2020 and full/limited range.
    """
    Yf = Y.astype(np.float32)
    Cb = U.astype(np.float32)
    Cr = V.astype(np.float32)

    m = (range_mode or "full").lower()
    if m == "limited":
        Yf = (Yf - 16.0) * (255.0 / 219.0)
        Cbf = (Cb - 128.0) * (255.0 / 224.0)
        Crf = (Cr - 128.0) * (255.0 / 224.0)
        Cb = Cbf + 128.0
        Cr = Crf + 128.0

    std = (standard or "bt601").lower()
    KRKGKB = {
        "bt601": (0.299000, 0.587000, 0.114000),
        "bt709": (0.212600, 0.715200, 0.072200),
        "bt2020": (0.262700, 0.678000, 0.059300),
    }
    if std not in KRKGKB:
        logger.warning(f"[hue] unknown standard '{standard}', fallback to bt601")
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
    rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    return rgb.astype(np.float32)


def _rgb_f32_to_yuv444(rgb: np.ndarray, standard: str = "bt601",
                       range_mode: str = "full") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB float32 in [0,1] → YUV444 uint8.
    Supports bt601/bt709/bt2020 and full/limited range.
    """
    R = rgb[..., 0] * 255.0
    G = rgb[..., 1] * 255.0
    B = rgb[..., 2] * 255.0

    std = (standard or "bt601").lower()
    KRKGKB = {
        "bt601": (0.299000, 0.587000, 0.114000),
        "bt709": (0.212600, 0.715200, 0.072200),
        "bt2020": (0.262700, 0.678000, 0.059300),
    }
    if std not in KRKGKB:
        std = "bt601"
    Kr, Kg, Kb = KRKGKB[std]

    Y = Kr * R + Kg * G + Kb * B
    Cb = (B - Y) / (2.0 * (1.0 - Kb))
    Cr = (R - Y) / (2.0 * (1.0 - Kr))

    U = Cb + 128.0
    V = Cr + 128.0

    m = (range_mode or "full").lower()
    if m == "limited":
        Y = 16.0 + Y * (219.0 / 255.0)
        U = 128.0 + (U - 128.0) * (224.0 / 255.0)
        V = 128.0 + (V - 128.0) * (224.0 / 255.0)
        Y = np.clip(Y, 16.0, 235.0)
        U = np.clip(U, 16.0, 240.0)
        V = np.clip(V, 16.0, 240.0)
    else:
        Y = np.clip(Y, 0.0, 255.0)
        U = np.clip(U, 0.0, 255.0)
        V = np.clip(V, 0.0, 255.0)

    return Y.astype(np.uint8), U.astype(np.uint8), V.astype(np.uint8)


# -------------------- Hue/Saturation in RGB --------------------
def hue_matrix(degrees: float) -> np.ndarray:
    """3x3 hue rotation matrix in RGB space (degrees)."""
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [0.213 + 0.787 * c - 0.213 * s, 0.715 - 0.715 * c - 0.715 * s, 0.072 - 0.072 * c + 0.928 * s],
        [0.213 - 0.213 * c + 0.143 * s, 0.715 + 0.285 * c + 0.140 * s, 0.072 - 0.072 * c - 0.283 * s],
        [0.213 - 0.213 * c - 0.787 * s, 0.715 - 0.715 * c + 0.715 * s, 0.072 + 0.928 * c + 0.072 * s],
    ], dtype=np.float32)


def saturate_matrix(s: float) -> np.ndarray:
    """3x3 saturation adjustment matrix (s=1 keeps chroma unchanged)."""
    return np.array([
        [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
        [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
        [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
    ], dtype=np.float32)


# -------------------- RGB → HSV (for selective tweaks) --------------------
def rgb_to_hsv_np(arr):
    """RGB [0,1] → HSV (H,S,V in [0,1]). Vectorized and branch-light."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 1e-8, delta / np.maximum(1e-8, maxc), 0.0)

    rc = (maxc - r) / np.maximum(delta, 1e-8)
    gc = (maxc - g) / np.maximum(delta, 1e-8)
    bc = (maxc - b) / np.maximum(delta, 1e-8)

    h = np.zeros_like(maxc)
    mask = delta > 1e-8
    rmax = mask & (r == maxc)
    gmax = mask & (g == maxc)
    bmax = mask & (b == maxc)

    h[rmax] = (bc - gc)[rmax]
    h[gmax] = 2.0 + (rc - bc)[gmax]
    h[bmax] = 4.0 + (gc - rc)[bmax]
    h = (h / 6.0) % 1.0

    return h, s, v


# -------------------- Core ops --------------------
def hue_sat_rgb(arr: np.ndarray, degrees: float, sat: float) -> np.ndarray:
    """
    Global hue rotation + saturation in RGB.
    Input/Output: (H,W,3) float32 in [0,1].
    """
    H = hue_matrix(degrees)
    S = saturate_matrix(sat)
    M = S @ H
    out = np.clip(arr @ M.T, 0.0, 1.0)
    return out


def selective_hue_rotate(arr, degrees, hue_min, hue_max, strength=1.0,
                         sigma_deg=None, v_min=0.4, v_max=1.0, s_min=0.08):
    """
    Selective hue rotation within a hue window; blended by hue/val/sat weights.
    degrees in RGB space; hue range in degrees; strength ∈ [0,1].
    """
    H = hue_matrix(degrees)
    arr_rot = np.clip(arr @ H.T, 0.0, 1.0)

    h, s, v = rgb_to_hsv_np(arr)
    h_deg = h * 360.0

    # Hue weight (Gaussian on circular distance)
    center = (hue_min + hue_max) * 0.5
    if sigma_deg is None:
        sigma_deg = (hue_max - hue_min) * 0.5
    dist = circ_dist_deg(h_deg, center)
    w_h = np.exp(-0.5 * (dist / np.maximum(1e-6, sigma_deg)) ** 2)

    # Value/Saturation weights
    w_v = smoothstep(v_min, v_max, v)
    w_s = smoothstep(s_min, 1.0, s)

    w = np.clip(w_h * w_v * w_s * strength, 0.0, 1.0)[..., None]
    out = np.clip(arr * (1.0 - w) + arr_rot * w, 0.0, 1.0)
    return out


def warm_highlights(arr, r_shift=0.04, b_shift=-0.03, v_lo=0.6, v_hi=0.9):
    """Warm highlights by boosting red and reducing blue in bright regions."""
    _, _, v = rgb_to_hsv_np(arr)
    w = smoothstep(v_lo, v_hi, v)[..., None]
    out = arr.copy()
    out[..., 0] = np.clip(out[..., 0] * (1.0 + r_shift * w[..., 0]), 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] * (1.0 + b_shift * w[..., 0]), 0.0, 1.0)
    return out


# -------------------- Combined pipeline --------------------
def process_rgb_hue_sat(
        rgb: np.ndarray,  # (H, W, 3) float32, 0-1
        deg_global: float = 0.0,
        sat_global: float = 1.0,
        do_selective: bool = False,
        sel_deg: float = 60.0,
        sel_hmin: float = 180.0,
        sel_hmax: float = 260.0,
        sel_strength: float = 0.8,
        sel_sigma: Optional[float] = None,
        sel_vmin: float = 0.4,
        sel_vmax: float = 1.0,
        sel_smin: float = 0.08,
        do_hi_warm: bool = False,
        hi_r: float = 0.04,
        hi_b: float = -0.03,
        hi_lo: float = 0.6,
        hi_hi: float = 0.9,
) -> np.ndarray:
    """
    Composite: global hue/sat → selective hue rotate → warm highlights.
    Returns RGB float32 in [0,1].
    """
    # 1) Global hue + saturation
    arr = hue_sat_rgb(rgb, deg_global, sat_global)

    # 2) Selective hue rotation
    if do_selective:
        arr = selective_hue_rotate(
            arr, sel_deg, sel_hmin, sel_hmax,
            strength=sel_strength, sigma_deg=sel_sigma,
            v_min=sel_vmin, v_max=sel_vmax, s_min=sel_smin
        )

    # 3) Warm highlights
    if do_hi_warm:
        arr = warm_highlights(arr, r_shift=hi_r, b_shift=hi_b, v_lo=hi_lo, v_hi=hi_hi)

    return np.clip(arr, 0.0, 1.0)


# -------------------- Preview helpers --------------------
def _yuv444_to_rgb_u8(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                      standard: str, range_mode: str) -> np.ndarray:
    """Preview helper: YUV → RGB uint8 with the selected matrix/range."""
    rgb_f = _yuv444_to_rgb_f32(Y, U, V, standard, range_mode)
    return (np.clip(rgb_f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _resize_max_side_uint8(img8: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Resize keeping aspect ratio so the longer side equals max_side."""
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
    """Save an RGB preview converted from YUV444 for quick inspection."""
    rgb = _yuv444_to_rgb_u8(Y, U, V, standard, range_mode)
    rgb = _resize_max_side_uint8(rgb, max_side)
    Image.fromarray(rgb, mode="RGB").save(out_path)
    return out_path


# -------------------- ISP entry --------------------
def run_hue_after_contrast(
        Y_in: np.ndarray,
        U_in: np.ndarray,
        V_in: np.ndarray,
        meta_in: Dict[str, Any],
        out_dir: str,
        # Global
        deg_global: float = 0.0,
        sat_global: float = 1.0,
        # Selective hue rotation
        enable_selective: bool = False,
        sel_deg: float = 60.0,
        sel_hue_range: Tuple[float, float] = (180.0, 260.0),
        sel_strength: float = 0.8,
        sel_sigma: Optional[float] = None,
        sel_v_range: Tuple[float, float] = (0.45, 1.0),
        sel_s_min: float = 0.08,
        # Warm highlights
        enable_hi_warm: bool = False,
        hi_r_shift: float = 0.04,
        hi_b_shift: float = -0.03,
        hi_v_range: Tuple[float, float] = (0.6, 0.9),
        # Saving
        range_mode: Optional[str] = None,
        standard: Optional[str] = None,
        save_raw: bool = True,
        save_png: bool = True,
        save_preview_rgb: bool = True,
        filename_stem: str = "yuv444_hue",
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """
    Hue/Saturation stage operating in RGB while I/O stays YUV444.
    Returns ((Y,U,V), meta_out, info).
    """
    _ensure_dir(out_dir)

    # Validate
    if not (isinstance(Y_in, np.ndarray) and isinstance(U_in, np.ndarray) and isinstance(V_in, np.ndarray)):
        raise TypeError("run_hue_after_contrast: Y/U/V must be numpy.ndarray")
    if not (Y_in.shape == U_in.shape == V_in.shape):
        raise ValueError("run_hue_after_contrast: shape mismatch")
    if Y_in.dtype != np.uint8:
        logger.warning("[hue] input is not uint8; will be interpreted as uint8")

    meta_out = dict(meta_in or {})
    std = (standard or meta_out.get("csc_standard") or "bt601")
    rng = (range_mode or meta_out.get("csc_range") or "full")

    info: Dict[str, Any] = {
        "deg_global": deg_global,
        "sat_global": sat_global,
        "enable_selective": enable_selective,
        "enable_hi_warm": enable_hi_warm,
        "standard": std,
        "range": rng,
        "saved": {}
    }

    # Preview before
    if save_preview_rgb:
        try:
            prev_before = Path(out_dir) / "preview_before_hue_yuv2rgb.png"
            _save_yuv_preview_rgb(Y_in, U_in, V_in, prev_before, std, rng)
            info["saved"]["preview_before"] = str(prev_before)
        except Exception as e:
            logger.warning("[hue] failed to save pre-preview: %s", e)

    # 1) YUV → RGB
    rgb = _yuv444_to_rgb_f32(Y_in, U_in, V_in, std, rng)

    # 2) RGB processing
    rgb_out = process_rgb_hue_sat(
        rgb,
        deg_global=deg_global,
        sat_global=sat_global,
        do_selective=enable_selective,
        sel_deg=sel_deg,
        sel_hmin=sel_hue_range[0],
        sel_hmax=sel_hue_range[1],
        sel_strength=sel_strength,
        sel_sigma=sel_sigma,
        sel_vmin=sel_v_range[0],
        sel_vmax=sel_v_range[1],
        sel_smin=sel_s_min,
        do_hi_warm=enable_hi_warm,
        hi_r=hi_r_shift,
        hi_b=hi_b_shift,
        hi_lo=hi_v_range[0],
        hi_hi=hi_v_range[1],
    )

    # 3) RGB → YUV
    Y_out, U_out, V_out = _rgb_f32_to_yuv444(rgb_out, std, rng)

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
            logger.warning("[hue] failed to save PNGs: %s", e)

    # Save planar .yuv
    if save_raw:
        try:
            yuv_path = Path(out_dir) / f"{filename_stem}.yuv"
            _save_planar_yuv444(str(yuv_path), Y_out, U_out, V_out)
            info["saved"]["raw"] = str(yuv_path)
        except Exception as e:
            logger.warning("[hue] failed to save planar .yuv: %s", e)

    # Preview after
    if save_preview_rgb:
        try:
            prev_after = Path(out_dir) / "preview_after_hue_yuv2rgb.png"
            _save_yuv_preview_rgb(Y_out, U_out, V_out, prev_after, std, rng)
            info["saved"]["preview_after"] = str(prev_after)
        except Exception as e:
            logger.warning("[hue] failed to save post-preview: %s", e)

    # Basic deltas
    try:
        dU = U_out.astype(np.float32) - U_in.astype(np.float32)
        dV = V_out.astype(np.float32) - V_in.astype(np.float32)
        info["delta_uv"] = {
            "u_abs_mean": float(np.mean(np.abs(dU))),
            "v_abs_mean": float(np.mean(np.abs(dV))),
            "u_abs_max": float(np.max(np.abs(dU))),
            "v_abs_max": float(np.max(np.abs(dV))),
        }
    except Exception:
        pass

    # Meta updates
    meta_out["hue_deg_global"] = deg_global
    meta_out["hue_sat_global"] = sat_global
    if enable_selective:
        meta_out["hue_selective"] = {
            "deg": sel_deg,
            "range": sel_hue_range,
            "strength": sel_strength
        }

    logger.info("[HUE] done | deg=%.1f sat=%.2f | selective=%s | hi_warm=%s",
                deg_global, sat_global, enable_selective, enable_hi_warm)

    return (Y_out, U_out, V_out), meta_out, info
