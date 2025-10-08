# core/tmc.py
# -*- coding: utf-8 -*-
"""
TMC (Tone Mapping on CSC Output)
--------------------------------
Runs after CSC (RGB -> YUV444, 8-bit). Applies tone mapping to Y' only; U/V are passed through.
Supports full/limited range and RGB preview using CSC user-coefficients, falling back to BT.601(JFIF).

Usage (example):
    (Y, U, V), meta_csc, _ = run_csc_rgb_to_yuv444(...)
    (Yt, Ut, Vt), meta_tmc, _ = run_tmc_after_csc(
        Y, U, V, meta_csc, out_dir="...", filename_stem="yuv444"
    )

Outputs:
    - <stem>_tmc.yuv        : planar 4:4:4 (Y plane | U plane | V plane)
    - <stem>_Y_tmc.png      : processed Y' (grayscale)
    - <stem>_preview_rgb.png (optional): 8-bit RGB preview (for visualization only)
"""
from __future__ import annotations
import os, json, math
from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tmc")

_EPS = 1e-8


# ========================= I/O myutils =========================

def _write_planar_yuv444(outfile: Path,
                         Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> Path:
    """Write planar YUV444 as Y, then U, then V."""
    outfile = Path(outfile)
    h, w = Y.shape
    assert U.shape == (h, w) and V.shape == (h, w), "Y/U/V shapes must match"
    with open(outfile, "wb") as f:
        f.write(Y.tobytes(order="C"))
        f.write(U.tobytes(order="C"))
        f.write(V.tobytes(order="C"))
    return outfile


def _save_png(img8: np.ndarray, path: Path, mode: str = "L") -> Path:
    """Save an 8-bit image with the given PIL mode."""
    Image.fromarray(img8, mode=mode).save(path)
    return Path(path)


# ========================= rounding/range =========================

def _round_half_up(x: np.ndarray) -> np.ndarray:
    """Round halves away from zero (avoid bankers' rounding)."""
    return np.floor(x + 0.5, dtype=np.float64)


def _range_to01_from_u8(y_u8: np.ndarray, range_mode: str) -> np.ndarray:
    """Map Y' (8-bit) to [0,1]. limited: 16..235 → 0..1; full: 0..255 → 0..1."""
    y = y_u8.astype(np.float32)
    rng = (range_mode or "full").lower()
    if rng == "limited":
        return np.clip((y - 16.0) / 219.0, 0.0, 1.0)
    return np.clip(y / 255.0, 0.0, 1.0)


def _range_from01_to_u8(y01: np.ndarray, range_mode: str) -> np.ndarray:
    """Map [0,1] back to Y' (8-bit). limited: 16..235; full: 0..255."""
    y01 = np.clip(y01, 0.0, 1.0).astype(np.float32)
    rng = (range_mode or "full").lower()
    if rng == "limited":
        y = 16.0 + 219.0 * y01
    else:
        y = 255.0 * y01
    y = _round_half_up(y).astype(np.int32)
    return np.clip(y, 0, 255).astype(np.uint8)


# ========================= stats/helpers =========================

def _safe_quantile(x: np.ndarray, q: float) -> float:
    """Quantile on finite values; returns 0 if empty."""
    x = np.asarray(x, np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    q = float(min(max(q, 0.0), 1.0))
    return float(np.quantile(x, q))


def _log_mean(x01: np.ndarray,
              lo_q: float = 0.01, hi_q: float = 0.99) -> float:
    """Log-average luminance proxy on clipped [lo_q, hi_q] range."""
    S = np.asarray(x01, np.float32)
    lo = _safe_quantile(S, lo_q)
    hi = _safe_quantile(S, hi_q)
    S = np.clip(S, lo, max(hi, lo + 1e-6))
    S = np.maximum(S, 0.0)
    pos = S > 0
    if not np.any(pos):
        return 1e-2
    return float(np.exp(np.mean(np.log(S[pos] + _EPS))))


def _smoothstep(a: float, b: float, x: np.ndarray) -> np.ndarray:
    """Smooth Hermite step from a to b on x."""
    t = (x - a) / max(b - a, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ========================= Filmic curves =========================

def _filmic_hable(x: np.ndarray) -> np.ndarray:
    """Hable filmic tone curve (unbounded in; clamped out)."""
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    num = x * (A * x + C * B) + D * E
    den = x * (A * x + B) + D * F
    y = num / (den + _EPS) - E / F
    return np.maximum(y, 0.0)


def _filmic_aces_approx(x: np.ndarray) -> np.ndarray:
    """ACES-like curve approximation in [0,1]."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e + _EPS), 0.0, 1.0)


# ========================= Lite local boost on Y' =========================

def _box_filter_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Simple box blur; radius 1 uses a fast 3x3 path."""
    if radius <= 0:
        return x
    x = np.asarray(x, np.float32)
    if radius == 1:
        xp = np.pad(x, ((1, 1), (1, 1)), mode="edge")
        s = (
            xp[0:-2, 0:-2] + xp[0:-2, 1:-1] + xp[0:-2, 2:] +
            xp[1:-1, 0:-2] + xp[1:-1, 1:-1] + xp[1:-1, 2:] +
            xp[2:, 0:-2] + xp[2:, 1:-1] + xp[2:, 2:]
        )
        return s / 9.0

    pad = int(radius)
    xp = np.pad(x, ((pad, pad), (pad, pad)), mode="edge").astype(np.float64, copy=False)
    I = np.cumsum(np.cumsum(np.pad(xp, ((1, 0), (1, 0)), mode="constant", constant_values=0.0),
                            axis=0), axis=1, dtype=np.float64)
    k = 2 * pad + 1
    out = I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]
    out = out / float(k * k)
    return out.astype(np.float32, copy=False)


def _local_lite_detail(L_in01: np.ndarray, radius: int, detail_pow: float) -> np.ndarray:
    """Edge-friendly local contrast: boost L/B ratio by detail_pow, then rescale."""
    B = _box_filter_2d(L_in01, max(int(radius), 0))
    B = np.maximum(B, _EPS)
    ratio = np.power(np.clip(L_in01 / B, 1e-3, 1e3), float(detail_pow))
    return np.clip(B * ratio, 0.0, 1.0)


# ========================= YUV -> RGB preview =========================

def _rgb_from_yuv_user_coeffs(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                              coeffs: Dict[str, list]) -> np.ndarray:
    """
    Invert user CSC coefficients (exact inverse of forward):
      Y = aR + bG + cB
      U-128 = dR + eG + fB
      V-128 = gR + hG + iB
    Solve for R,G,B and clip to 8-bit.
    """
    a, b, c = map(float, coeffs.get("Y", [0.299, 0.587, 0.114]))
    du = coeffs.get("U", [-0.14713, -0.28886, 0.436, 128.0])
    dv = coeffs.get("V", [0.615, -0.51498, -0.10001, 128.0])
    d, e, f, u_off = float(du[0]), float(du[1]), float(du[2]), float(du[3])
    g, h, i, v_off = float(dv[0]), float(dv[1]), float(dv[2]), float(dv[3])

    Yf = Y.astype(np.float32)
    Uf = U.astype(np.float32) - u_off
    Vf = V.astype(np.float32) - v_off

    M = np.array([[a, b, c],
                  [d, e, f],
                  [g, h, i]], dtype=np.float32)
    Mi = np.linalg.inv(M).astype(np.float32)

    h, w = Y.shape
    YUV = np.stack([Yf, Uf, Vf], axis=-1).reshape(-1, 3)
    RGB = (YUV @ Mi.T).reshape(h, w, 3)

    RGB = _round_half_up(RGB).astype(np.int32)
    RGB = np.clip(RGB, 0, 255).astype(np.uint8)
    return RGB


def _rgb_from_ycbcr601_full(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    BT.601 (JFIF, full-range) inverse for preview:
      R' = Y' + 1.40200*(Cr-128)
      G' = Y' - 0.344136*(Cb-128) - 0.714136*(Cr-128)
      B' = Y' + 1.77200*(Cb-128)
    """
    Yf = Y.astype(np.float32)
    Cbf = Cb.astype(np.float32) - 128.0
    Crf = Cr.astype(np.float32) - 128.0

    R = Yf + 1.40200 * Crf
    G = Yf - 0.344136 * Cbf - 0.714136 * Crf
    B = Yf + 1.77200 * Cbf

    RGB = np.stack([R, G, B], axis=-1)
    RGB = _round_half_up(RGB).astype(np.int32)
    RGB = np.clip(RGB, 0, 255).astype(np.uint8)
    return RGB


# ========================= Main: run after CSC =========================

def run_tmc_after_csc(
    Y_in: np.ndarray,
    U_in: np.ndarray,
    V_in: np.ndarray,
    meta_in: Dict,
    out_dir: str,
    *,
    range_mode: Optional[str] = None,      # Override meta["range"] ("full"/"limited")
    curve: Optional[str] = None,           # "aces" | "hable" | None(auto)
    enable_local_lite: bool = True,        # Lite local contrast
    local_radius: int = 8,
    local_detail_pow: float = 1.02,
    save_preview_rgb: bool = True,
    filename_stem: str = "yuv444"
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict, Dict]:
    """
    Tone-map Y' only (global filmic + optional local-lite). U/V are unchanged.
    Returns: ((Y_tmc, U_in, V_in), meta_out, info)
    """
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if not (isinstance(Y_in, np.ndarray) and isinstance(U_in, np.ndarray) and isinstance(V_in, np.ndarray)):
        raise TypeError("run_tmc_after_csc: Y/U/V must be ndarray")

    if Y_in.dtype != np.uint8 or U_in.dtype != np.uint8 or V_in.dtype != np.uint8:
        raise TypeError("run_tmc_after_csc: expected uint8 inputs")

    if Y_in.shape != U_in.shape or Y_in.shape != V_in.shape:
        raise ValueError("run_tmc_after_csc: Y/U/V sizes must match")

    rng = (range_mode or meta_in.get("range") or "full").lower()
    rng = "limited" if rng.startswith("lim") else "full"

    # ---------- 1) Y' → [0,1] + quick stats ---------- #
    Y01 = _range_to01_from_u8(Y_in, rng)
    q01 = _safe_quantile(Y01, 0.01)
    q50 = _safe_quantile(Y01, 0.50)
    q95 = _safe_quantile(Y01, 0.95)
    q99 = _safe_quantile(Y01, 0.99)
    DR  = float(np.log2((q99 + 1e-6) / (q01 + 1e-6)))  # approx dynamic range
    log_avg = _log_mean(Y01, 0.01, 0.99)

    info = dict(
        q01=float(q01), q50=float(q50), q95=float(q95), q99=float(q99),
        DR_log2=float(DR), log_avg=float(log_avg), range=rng
    )

    # ---------- 2) Auto exposure toward middle gray ---------- #
    # Note: Y' is not linear; this heuristic is sufficient for stable perception.
    if log_avg < 0.10:   middle_gray = 0.20
    elif log_avg < 0.16: middle_gray = 0.19
    elif log_avg > 0.30: middle_gray = 0.16
    elif log_avg > 0.22: middle_gray = 0.17
    else:                 middle_gray = 0.18

    exposure_raw = (middle_gray + _EPS) / (log_avg + _EPS)
    exp_min, exp_max = 0.25, (10.0 if DR >= 5.0 else 12.0)
    exposure = float(np.clip(exposure_raw, exp_min, exp_max))
    info.update(middle_gray=float(middle_gray), exposure=float(exposure),
                exposure_raw=float(exposure_raw), exp_limits=(exp_min, exp_max))

    L1 = Y01 * exposure

    # ---------- 3) Filmic curve ---------- #
    if curve is None:
        curve_sel = "hable" if (DR >= 6.0 or (Y01 > q95).mean() >= 0.06) else "aces"
    else:
        curve_sel = str(curve).lower()
    filmic = _filmic_aces_approx if curve_sel == "aces" else _filmic_hable
    L2 = filmic(L1)

    # ---------- 4) White normalization ---------- #
    p_w = _safe_quantile(L2, 0.989 if DR > 5.0 else 0.995)
    scale_w_max = 3.2 if DR >= 4.5 else 4.0
    p_floor = 1.0 / max(scale_w_max, 1e-6)
    p_w_safe = max(p_w, p_floor)
    scale_w = float(min(1.0 / max(p_w_safe, _EPS), scale_w_max))
    L3 = np.clip(L2 * scale_w, 0.0, 1.0)
    info.update(curve=curve_sel, p_white=float(p_w),
                white_scale=float(scale_w), white_scale_max=float(scale_w_max))

    # ---------- 5) Lite local enhancement on Y' ---------- #
    if enable_local_lite:
        Loc = _local_lite_detail(L3, radius=int(local_radius), detail_pow=float(local_detail_pow))
        # Clamp residual to avoid halos/noise inflation
        delta = np.clip(Loc - L3, -0.10, 0.12 if DR < 6.0 else 0.10)
        L4 = np.clip(L3 + delta, 0.0, 1.0)
        info.update(local_lite=True, local_radius=int(local_radius),
                    local_detail_pow=float(local_detail_pow))
    else:
        L4 = L3
        info.update(local_lite=False)

    # ---------- 6) Write back Y' (U/V passthrough) ---------- #
    Y_out = _range_from01_to_u8(L4, rng)
    U_out, V_out = U_in, V_in

    # ---------- 7) Save outputs ---------- #
    stem = filename_stem or "yuv444"
    saves = {}

    y_path = _save_png(Y_out, out_p / f"{stem}_Y_tmc.png", mode="L")
    saves["Y_tmc_png"] = str(y_path)

    raw_path = _write_planar_yuv444(out_p / f"{stem}_tmc.yuv", Y_out, U_out, V_out)
    saves["yuv444_tmc"] = str(raw_path)

    # Optional RGB preview (debug/visualization only)
    if save_preview_rgb:
        try:
            coeffs = None
            if isinstance(meta_in, dict):
                coeffs = meta_in.get("coeffs") or meta_in.get("csc_coeffs")
            if isinstance(coeffs, dict):
                rgb8 = _rgb_from_yuv_user_coeffs(Y_out, U_out, V_out, coeffs)
            else:
                rgb8 = _rgb_from_ycbcr601_full(Y_out, U_out, V_out)
            rgb_path = _save_png(rgb8, out_p / f"{stem}_preview_rgb.png", mode="RGB")
            saves["preview_rgb_png"] = str(rgb_path)
        except Exception as e:
            logger.warning("[TMC] RGB preview failed: %s (does not affect main flow)", e)

    # ---------- 8) meta / info ---------- #
    meta_out = dict(meta_in)
    meta_out.update({
        "tmc_after": "csc",
        "tmc_curve": curve_sel,
        "tmc_range": rng,
        "yuv_format": meta_in.get("yuv_format", "444"),
        "bitdepth": 8
    })
    info["saved"] = saves

    logger.info("[TMC] Y’ tone mapping done: saved=%s", json.dumps(saves, ensure_ascii=False))
    return (Y_out, U_out, V_out), meta_out, info


# ========================= Option: run from planar file =========================

def run_tmc_from_planar_file(
    yuv_path: str, width: int, height: int, meta_in: Dict, out_dir: str,
    **kwargs
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict, Dict]:
    """
    Convenience path when you only have a planar .yuv and dimensions:
        (Yt, Ut, Vt), meta_out, info = run_tmc_from_planar_file(
            "yuv444.yuv", W, H, meta_csc, out_dir="..."
        )
    """
    p = Path(yuv_path)
    nbytes = width * height
    with open(p, "rb") as f:
        buf = f.read(nbytes * 3)
    if len(buf) < nbytes * 3:
        raise IOError("file size does not match width/height")

    Y = np.frombuffer(buf[0:nbytes], dtype=np.uint8).reshape(height, width)
    U = np.frombuffer(buf[nbytes:2*nbytes], dtype=np.uint8).reshape(height, width)
    V = np.frombuffer(buf[2*nbytes:3*nbytes], dtype=np.uint8).reshape(height, width)
    return run_tmc_after_csc(Y, U, V, meta_in, out_dir, **kwargs)
