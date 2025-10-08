# core/awb.py
# -*- coding: utf-8 -*-
"""
Production AWB in the Bayer/RAW domain, used in the pipeline:
- API matches run_bm3d_raw: returns (raw_out_lin, meta_out, info)
- Placed after BM3D; input must be BLC'ed linear RAW
- Strategy: per-channel Top% white estimation
- Stats domain uses (value - black_level_after_blc).relu()
- Thresholds given as fractions of white_level and mapped to the stats domain
- Outputs previews (2x2 block RGB) and saves gains as JSON/NPZ
"""
from __future__ import annotations
import os, json
from typing import Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections.abc import Mapping

logger = logging.getLogger("core.awb")

# ---------------- Common: fold different "levels" to a scalar ---------------- #
def _as_scalar_level(val, default: float = 0.0) -> float:
    """Reduce scalar/list/ndarray/dict into a single scalar via mean; fall back to default on failure."""
    try:
        if val is None:
            return float(default)
        if np.isscalar(val):
            return float(val)
        if isinstance(val, Mapping):
            flat = []
            for v in val.values():
                if v is None: continue
                if np.isscalar(v):
                    flat.append(float(v))
                else:
                    arr = np.array(v, dtype=np.float64).ravel()
                    if arr.size: flat.append(float(arr.mean()))
            return float(np.mean(flat)) if flat else float(default)
        arr = np.array(val, dtype=np.float64).ravel()
        return float(arr.mean()) if arr.size else float(default)
    except Exception:
        return float(default)

# ---------------- Bayer mapping & small helpers ---------------- #
def _get_bayer_slices(h, w, pattern: str):
    """
    Return slices for four Bayer planes (R/G1/G2/B) on a 2D RAW image.
    Uses 2x2 stepping (even/odd rows/cols). Returns dict of (row_slice, col_slice).
    """
    p = (pattern or "RGGB").upper()
    if p not in ("RGGB", "GRBG", "GBRG", "BGGR"):
        raise ValueError(f"[AWB] 不支持的 Bayer 模式：{pattern}")
    rr0, rr1 = slice(0, None, 2), slice(1, None, 2)
    cc0, cc1 = slice(0, None, 2), slice(1, None, 2)
    if p == "RGGB": return {"R": (rr0, cc0), "G1": (rr0, cc1), "G2": (rr1, cc0), "B": (rr1, cc1)}
    if p == "GRBG": return {"R": (rr0, cc1), "G1": (rr0, cc0), "G2": (rr1, cc1), "B": (rr1, cc0)}
    if p == "GBRG": return {"R": (rr1, cc0), "G1": (rr0, cc0), "G2": (rr1, cc1), "B": (rr0, cc1)}
    if p == "BGGR": return {"R": (rr1, cc1), "G1": (rr0, cc1), "G2": (rr1, cc0), "B": (rr0, cc0)}

def _choose_adaptive_top_percent(
    R, G1, G2, B,
    black_level: float,
    white_level: float,
    *,
    min_p: float = 0.1,
    max_p: float = 14.0
) -> float:
    """
    Pick a Top% (as percentage, not 0~1) adaptively for robust white estimation.
    Uses the (value - black).relu() domain and considers exposure, saturation ratio, and bright-end spread.
    """

    # ---- Stats domain: (value - black).relu() ----
    Rf  = np.maximum(R.astype(np.float32)  - black_level, 0.0)
    G1f = np.maximum(G1.astype(np.float32) - black_level, 0.0)
    G2f = np.maximum(G2.astype(np.float32) - black_level, 0.0)
    Bf  = np.maximum(B.astype(np.float32)  - black_level, 0.0)
    Gf = 0.5 * (G1f + G2f)                # average greens for stability

    wl = max(float(white_level), 1.0)     # guard against zero

    # ---- Metric 1: exposure via median(G)/WL ----
    exp = float(np.median(Gf) / wl)

    # ---- Metric 2: near-saturation ratio @ 99.5% WL (any plane) ----
    sat_thr = wl * 0.995
    sat_ratio = float(np.mean(
        (Rf  >= sat_thr) |
        (G1f >= sat_thr) |
        (G2f >= sat_thr) |
        (Bf  >= sat_thr)
    ))

    # ---- Metric 3: bright-end tension (P99-P50)/WL ----
    if Gf.size:
        p50, p99 = np.percentile(Gf, [50, 99])
    else:
        p50, p99 = 0.0, 0.0
    dr_norm = float(max(p99 - p50, 0.0) / wl)

    # ---- Baseline Top% by exposure tiers ----
    if   exp < 0.08:  p = 14.0
    elif exp < 0.15:  p = 12.0
    elif exp < 0.35:  p = 4.5
    elif exp < 0.60:  p = 2.0
    else:             p = 0.5

    # ---- Shrink if more saturation ----
    if   sat_ratio > 0.05: p *= 0.5
    elif sat_ratio > 0.01: p *= 0.8

    # ---- Adjust by bright-end spread; slightly increase when very dark and flat ----
    if dr_norm > 0.30:
        p *= 0.7
    elif dr_norm < 0.05 and exp < 0.12:
        p *= 1.3

    # ---- Clamp and return (percentage) ----
    return float(np.clip(p, min_p, max_p))

def _simple_awb_bias(rgain: float, bgain: float,
                     *,
                     warm: float = 0.06,     # warm bias: +R and -B
                     max_ratio: float = 1.35 # cap B relative to R
                     ) -> tuple[float, float]:
    """Apply a small warm bias and guardrails to avoid overly cool results."""
    # 1) small warm shift
    rgain *= (1.0 + warm)
    bgain *= (1.0 - warm)

    # 2) relative clamp to keep B from dominating
    if bgain > rgain * max_ratio:
        bgain = rgain * max_ratio

    # 3) absolute clamp (conservative range)
    rgain = float(np.clip(rgain, 0.5, 3.0))
    bgain = float(np.clip(bgain, 0.5, 3.0))
    return rgain, bgain



def _top_percent_channel_means_rgb(R, G1, G2, B, percent=1.0,
                                   black_level=0.0, lower_thresh=None, upper_thresh=None):
    """
    Per-channel Top% statistics in the (value - black).relu() domain with optional low/high gating.
    Returns {R: r_mean, G_avg: (g1+g2)/2, B: b_mean}.
    """
    # Normalize to stats domain
    Rf  = np.maximum(R.astype(np.float32)  - black_level, 0.0)
    G1f = np.maximum(G1.astype(np.float32) - black_level, 0.0)
    G2f = np.maximum(G2.astype(np.float32) - black_level, 0.0)
    Bf  = np.maximum(B.astype(np.float32)  - black_level, 0.0)

    # Optional gates to drop too dark/near-saturated samples
    if lower_thresh is not None:
        Rf = Rf[Rf > lower_thresh]; G1f = G1f[G1f > lower_thresh]
        G2f = G2f[G2f > lower_thresh]; Bf = Bf[Bf > lower_thresh]
    if upper_thresh is not None:
        Rf = Rf[Rf < upper_thresh]; G1f = G1f[G1f < upper_thresh]
        G2f = G2f[G2f < upper_thresh]; Bf = Bf[Bf < upper_thresh]

    def top_mean(x, p):
        # Mean of the brightest p% in one channel
        if x.size == 0 or p <= 0: return float("nan")
        thr = np.percentile(x, 100 - p)
        return float(x[x >= thr].mean())

    r_mean  = top_mean(Rf,  percent)
    g1_mean = top_mean(G1f, percent)
    g2_mean = top_mean(G2f, percent)
    b_mean  = top_mean(Bf,  percent)
    return {"R": r_mean, "G_avg": 0.5 * (g1_mean + g2_mean), "B": b_mean}

def _bayer_channel_means_all(x: np.ndarray, slices, black_level: float = 0.0):
    """
    Compute global means per Bayer plane in the (value - black) domain; used for logging/fallback.
    """
    xr  = x[slices["R"]].astype(np.float32)  - black_level
    xg1 = x[slices["G1"]].astype(np.float32) - black_level
    xg2 = x[slices["G2"]].astype(np.float32) - black_level
    xb  = x[slices["B"]].astype(np.float32)  - black_level
    return {
        "R": float(xr.mean()),
        "G_avg": 0.5 * (float(xg1.mean()) + float(xg2.mean())),
        "B": float(xb.mean())
    }

def _downsample_bayer_to_rgb(raw_lin: np.ndarray, pattern: str) -> np.ndarray:
    """
    Preview-only: 2x2 block downsample from Bayer RAW to (H/2, W/2, 3) linear RGB.
    Assumes the input is in a consistent linear scale (e.g., normalized to [0,1]).
    """
    h, w = raw_lin.shape
    sl = _get_bayer_slices(h, w, pattern)
    R  = raw_lin[sl["R"]]
    G1 = raw_lin[sl["G1"]]
    G2 = raw_lin[sl["G2"]]
    B  = raw_lin[sl["B"]]
    G  = 0.5 * (G1 + G2)
    return np.stack([R, G, B], axis=-1)

# --- Linear → sRGB and simple AE for preview --- #
def _lin_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear [0,1] → sRGB OETF (for preview PNG saving only)."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    a = 0.055; thr = 0.0031308
    out = np.empty_like(x)
    m = x <= thr
    out[m]  = 12.92 * x[m]
    out[~m] = (1 + a) * np.power(x[~m], 1.0 / 2.4) - a
    return out

def _auto_expose_for_preview(rgb_lin: np.ndarray, percentile: float = 99.5,
                             target: float = 0.90, max_gain: float = 8.0) -> np.ndarray:
    """
    Simple preview AE: map luminance P{percentile} to a target level with a gain cap.
    Helps avoid overly dark/bright previews when inspecting AWB.
    """
    Y = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    if Y.size == 0: return rgb_lin
    p = float(np.percentile(Y, percentile))
    if p > 1e-6:
        gain = min(target / p, max_gain)
        rgb_lin = np.clip(rgb_lin * gain, 0.0, 1.0)
    return rgb_lin

# ---------------- Gain computation & main flow ---------------- #
def _compute_gains(R, G1, G2, B, percent, black_level, lower_t, upper_t):
    """
    Estimate representative bright-end means via per-channel Top%, then compute gains vs G.
    Falls back to global means on empty/NaN stats and clamps gains to a safe range.
    """
    stats = _top_percent_channel_means_rgb(R, G1, G2, B, percent=percent, black_level=black_level,
                                           lower_thresh=lower_t, upper_thresh=upper_t)
    r_mean = stats["R"]; g_mean = stats["G_avg"]; b_mean = stats["B"]
    if not np.isfinite(r_mean) or not np.isfinite(g_mean) or not np.isfinite(b_mean):
        # Fallback: global means in the relu domain
        Rf  = np.maximum(R.astype(np.float32)  - black_level, 0.0)
        G1f = np.maximum(G1.astype(np.float32) - black_level, 0.0)
        G2f = np.maximum(G2.astype(np.float32) - black_level, 0.0)
        Bf  = np.maximum(B.astype(np.float32)  - black_level, 0.0)
        r_mean = float(Rf.mean())
        g_mean = 0.5 * (float(G1f.mean()) + float(G2f.mean()))
        b_mean = float(Bf.mean())
    eps = 1e-6
    rgain = float(np.clip(g_mean / max(r_mean, eps), 0.2, 8.0))
    bgain = float(np.clip(g_mean / max(b_mean, eps), 0.2, 8.0))
    return rgain, 1.0, bgain

def run_awb_raw(raw_in_lin: np.ndarray,
                meta_in: Dict[str, Any],
                out_dir: str,
                *,
                top_percent: str | float = "auto",
                lower_frac: float = 0.03,
                upper_frac: float = 0.95,
                save_plots: bool = True) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Main AWB entry after BM3D. Input must be 2D linear Bayer RAW.
    Returns (corrected RAW, updated meta, info dict).
    """
    assert raw_in_lin.ndim == 2, "[AWB] 输入必须是二维 Bayer RAW"
    os.makedirs(out_dir, exist_ok=True)

    h, w = raw_in_lin.shape
    pattern = (meta_in.get("bayer") or "RGGB").upper()

    # === Fetch white_level / black_level_after_blc as scalars === #
    wl_default = np.iinfo(raw_in_lin.dtype).max if np.issubdtype(raw_in_lin.dtype, np.integer) else 1.0
    wl = _as_scalar_level(meta_in.get("white_level", None), default=float(wl_default))

    # BLC sets black_level_after_blc≈0; do not fall back to original black_level
    bl_after = _as_scalar_level(meta_in.get("black_level_after_blc", None), default=0.0)

    sl = _get_bayer_slices(h, w, pattern)

    # === Map WL-fraction thresholds into the relu domain (drop too dark/bright) === #
    lower_t = max(lower_frac * wl - bl_after, 0.0) if lower_frac is not None else None
    upper_t = max(upper_frac * wl - bl_after, 0.0) if upper_frac is not None else None

    # Channel views (read-only)
    x = raw_in_lin.astype(np.float32, copy=False)
    R, G1, G2, B = x[sl["R"]], x[sl["G1"]], x[sl["G2"]], x[sl["B"]]

    # === Adaptive Top% or user-specified value === #
    is_adaptive = (isinstance(top_percent, str) and top_percent.lower() in ("auto", "adaptive")) \
                  or (isinstance(top_percent, (int, float)) and top_percent <= 0)
    try:
        eff_top = float(top_percent) if not is_adaptive else _choose_adaptive_top_percent(R, G1, G2, B, bl_after, wl)
    except Exception:
        eff_top = 1.0  # safe fallback

    # Pre stats (global means) for logging/info
    pre_all = _bayer_channel_means_all(x, sl, black_level=bl_after)

    # === Compute gains (Top% per-channel; G as reference) === #
    rgain, ggain, bgain = _compute_gains(R, G1, G2, B, eff_top, bl_after, lower_t, upper_t)

    # Small warm bias + guardrails to avoid overly cool results
    rgain, bgain = _simple_awb_bias(rgain, bgain, warm=0.07, max_ratio=1.20)

    gains4 = [rgain, ggain, bgain, ggain]  # order: R, G1, B, G2 (downstream may assume RGGB order)

    # === Apply gains and clamp to WL (keep RAW units) === #
    y = x.copy()
    y[sl["R"]]  *= rgain
    y[sl["G1"]] *= ggain
    y[sl["G2"]] *= ggain
    y[sl["B"]]  *= bgain
    y = np.clip(y, 0.0, wl)

    post_all = _bayer_channel_means_all(y, sl, black_level=bl_after)

    # === Save previews (2x2 downsample → sRGB) === #
    if save_plots:
        denom = max(wl - bl_after, 1e-6)  # normalize the relu domain to [0,1]
        rgb_before_lin = _downsample_bayer_to_rgb(np.clip(x - bl_after, 0.0, None) / denom, pattern)
        rgb_after_lin  = _downsample_bayer_to_rgb(np.clip(y - bl_after, 0.0, None) / denom, pattern)
        # rgb_before_lin = _auto_expose_for_preview(rgb_before_lin)
        # rgb_after_lin  = _auto_expose_for_preview(rgb_after_lin)
        plt.imsave(os.path.join(out_dir, "awb_preview_before.png"), _lin_to_srgb(rgb_before_lin))
        plt.imsave(os.path.join(out_dir, "awb_preview_after.png"),  _lin_to_srgb(rgb_after_lin))

    # === Persist gains/stats === #
    info: Dict[str, Any] = {
        "eff_top_percent": float(eff_top),
        "gains": gains4,
        "pre_means_all": pre_all,
        "post_means_all": post_all,
        "lower_thresh": None if lower_t is None else float(lower_t),
        "upper_thresh": None if upper_t is None else float(upper_t),
        "pattern": pattern,
        "white_level_scalar": float(wl),
        "black_level_after_blc_used": float(bl_after),
    }
    try:
        with open(os.path.join(out_dir, "awb_gains.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        np.savez(os.path.join(out_dir, "awb_gains.npz"),
                 **{k: np.array([v], dtype=object) for k, v in info.items()})
        np.save(os.path.join(out_dir, "awb_corrected_raw.npy"), y.astype(raw_in_lin.dtype, copy=False))
    except Exception as e:
        logger.warning("[AWB] 保存文件失败：%s", e)

    # === Update meta to mark WB applied (so downstream won't redo WB) === #
    meta_out = dict(meta_in)
    meta_out["awb_gains"] = gains4
    meta_out["awb_top_percent"] = float(eff_top)
    meta_out["as_shot_neutral"] = [1.0, 1.0, 1.0]
    meta_out["wb_applied"] = "raw_awb_linear"

    logger.info("[AWB] eff_top=%.3f%% | gains: R=%.4f G=%.4f B=%.4f | pre(R,G,B)=(%.2f,%.2f,%.2f) → post(%.2f,%.2f,%.2f) | wl=%.1f bl_after=%.3f",
                eff_top, gains4[0], gains4[1], gains4[2],
                pre_all["R"], pre_all["G_avg"], pre_all["B"],
                post_all["R"], post_all["G_avg"], post_all["B"],
                wl, bl_after)

    return y.astype(raw_in_lin.dtype, copy=False), meta_out, info
