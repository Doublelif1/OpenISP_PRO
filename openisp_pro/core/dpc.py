# -*- coding: utf-8 -*-
"""
Dead Pixel Correction (DPC) — production version (color-plane aware + friendly previews + adaptive threshold)

New features:
- thres_mode in {'abs','rel','auto'}; default 'auto'
- auto: estimate noise via same-color neighbor MAD → σ, threshold = auto_k * σ (default 6.0)
- rel: treat dpc_thres as a ratio of WL (e.g., 0.006 = 0.6% of WL)
- “≥M/8 outliers” rule for bad pixels (bad_neighbor_min, default 8; use 6 to reduce misses)
- Everything else stays compatible (two previews; returns corrected_raw and bad_mask)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# =========================== Optional: Numba acceleration ============================
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# Matplotlib font (so Chinese filenames/labels render safely if present)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== Preview tone-map params (affect PNG previews only; not the numeric results) =====
_PREVIEW_LOW_Q   = 0.001
_PREVIEW_HIGH_Q  = 0.999
_PREVIEW_GAMMA   = 2.2
_PREVIEW_EPS     = 1e-12


def _normalize_to_01(img, wl):
    """Map to [0,1] by WL (utility for previews)."""
    arr = img.astype(np.float64)
    return np.clip(arr / max(int(wl) if wl is not None else 1, 1), 0.0, 1.0)


def _extract_black_level(meta):
    """Get an approximate scalar black level from meta (avg if 4-channel dict)."""
    if meta is None:
        return 0.0
    bl = meta.get("black_level", 0.0)
    try:
        if isinstance(bl, dict):
            vals = [float(v) for v in bl.values() if v is not None]
            return float(np.mean(vals)) if vals else 0.0
        elif isinstance(bl, (list, tuple, np.ndarray)):
            arr = np.asarray(bl, dtype=np.float64)
            return float(np.nanmean(arr)) if arr.size else 0.0
        else:
            return float(bl)
    except Exception:
        return 0.0


def _preview_map(img, wl, bl=None, low_q=_PREVIEW_LOW_Q, high_q=_PREVIEW_HIGH_Q, gamma=_PREVIEW_GAMMA):
    """Build a simple percentile-based tone map with optional BL offset and gamma."""
    x = img.astype(np.float64, copy=False)
    wl = float(wl if wl is not None else (np.max(x) if x.size else 1.0))
    wl = max(wl, 1.0)
    bl = 0.0 if bl is None else float(bl)

    y = np.clip(x - bl, 0.0, wl)

    if y.size:
        vmin = float(np.quantile(y, low_q))
        vmax = float(np.quantile(y, high_q))
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = wl
    if vmax - vmin < 1e-9:
        vmin, vmax = 0.0, wl

    z = (y - vmin) / max(vmax - vmin, _PREVIEW_EPS)
    z = np.clip(z, 0.0, 1.0)
    if gamma and gamma > 0:
        z = np.power(z, 1.0 / float(gamma))
    return z


def _save_gray_png_preview(path, img, wl, meta=None):
    """Save a grayscale preview PNG using percentile mapping (for quick QA)."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    bl = _extract_black_level(meta)
    arr01 = _preview_map(img, wl, bl=bl)
    plt.imsave(str(path), arr01, cmap='gray', vmin=0, vmax=1)
    logger.info("[DPC preview saved] %s (WL=%s, BL≈%.2f)", path, str(wl), bl)


def _bayer_to_lut(bayer: str) -> np.ndarray:
    """Map Bayer pattern to a 2×2 LUT: 0=R, 1=G, 2=B."""
    s = (bayer or "RGGB").upper()
    if s not in {"RGGB", "GRBG", "GBRG", "BGGR"}:
        logger.warning("[DPC] unknown bayer=%s, fallback to RGGB", s)
        s = "RGGB"
    grid = [[s[0], s[1]], [s[2], s[3]]]
    lut = np.zeros((2, 2), dtype=np.uint8)
    for i in range(2):
        for j in range(2):
            c = grid[i][j]
            lut[i, j] = 0 if c == 'R' else (1 if c == 'G' else 2)
    return lut


def _estimate_sigma_samecolor(img: np.ndarray, bayer: str = "RGGB") -> float:
    """
    Robust same-color noise sigma:
    - Use opposite-direction same-color differences: |q2-q7|, |q4-q5|, |q1-q8|, |q3-q6|
    - MAD(diff)/0.6745 ≈ σ_diff; for two independent samples, σ ≈ σ_diff/√2.
    """
    h, w = img.shape
    if h < 6 or w < 6:
        return 10.0  # Small image fallback

    bayer = (bayer or "RGGB").upper()
    lut = _bayer_to_lut(bayer)
    pad = np.pad(img, (2, 2), 'reflect').astype(np.int64)

    diffs = []

    # Subsample every 2 pixels to reduce cost.
    for y in range(0, h, 2):
        py = y + 2
        for x in range(0, w, 2):
            px = x + 2
            cid = int(lut[y & 1, x & 1])  # 0=R,1=G,2=B

            p1 = int(pad[py - 2, px - 2]); p2 = int(pad[py - 2, px    ]); p3 = int(pad[py - 2, px + 2])
            p4 = int(pad[py    , px - 2]);                                  p5 = int(pad[py    , px + 2])
            p6 = int(pad[py + 2, px - 2]); p7 = int(pad[py + 2, px    ]); p8 = int(pad[py + 2, px + 2])

            if cid == 1:  # G uses unit diagonals for same-color neighbors
                q1 = int(pad[py - 1, px - 1]); q2 = p2;                       q3 = int(pad[py - 1, px + 1])
                q4 = p4;                                                      q5 = p5
                q6 = int(pad[py + 1, px - 1]); q7 = p7;                       q8 = int(pad[py + 1, px + 1])
            else:
                q1, q2, q3, q4, q5, q6, q7, q8 = p1, p2, p3, p4, p5, p6, p7, p8

            diffs.append(abs(q2 - q7))
            diffs.append(abs(q4 - q5))
            diffs.append(abs(q1 - q8))
            diffs.append(abs(q3 - q6))

    if not diffs:
        return 10.0
    diffs = np.asarray(diffs, dtype=np.float64)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    if mad <= 1e-9:
        return 10.0
    sigma_diff = mad / 0.6745
    sigma = sigma_diff / np.sqrt(2.0)
    return float(max(sigma, 1.0))  # Floor for stability


# -------------------- Optional Numba kernels --------------------
if '_HAS_NUMBA' in globals() and _HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _dpc_core_njit(img_pad, h, w, thres, mode_id, clipv, color_lut, bad_neighbor_min):
        out = np.empty((h, w), dtype=img_pad.dtype)
        msk = np.zeros((h, w), dtype=np.uint8)

        for y in prange(h):
            py = y + 2
            for x in range(w):
                px = x + 2
                p0 = int(img_pad[py, px])
                cid = int(color_lut[y & 1, x & 1])  # 0=R,1=G,2=B

                p1 = int(img_pad[py - 2, px - 2]); p2 = int(img_pad[py - 2, px    ]); p3 = int(img_pad[py - 2, px + 2])
                p4 = int(img_pad[py    , px - 2]);                                      p5 = int(img_pad[py    , px + 2])
                p6 = int(img_pad[py + 2, px - 2]); p7 = int(img_pad[py + 2, px    ]); p8 = int(img_pad[py + 2, px + 2])

                if cid == 1:
                    q1 = int(img_pad[py - 1, px - 1]); q2 = p2;                         q3 = int(img_pad[py - 1, px + 1])
                    q4 = p4;                                                            q5 = p5
                    q6 = int(img_pad[py + 1, px - 1]); q7 = p7;                         q8 = int(img_pad[py + 1, px + 1])
                else:
                    q1, q2, q3, q4, q5, q6, q7, q8 = p1, p2, p3, p4, p5, p6, p7, p8

                cnt = 0
                if abs(q1 - p0) > thres: cnt += 1
                if abs(q2 - p0) > thres: cnt += 1
                if abs(q3 - p0) > thres: cnt += 1
                if abs(q4 - p0) > thres: cnt += 1
                if abs(q5 - p0) > thres: cnt += 1
                if abs(q6 - p0) > thres: cnt += 1
                if abs(q7 - p0) > thres: cnt += 1
                if abs(q8 - p0) > thres: cnt += 1

                is_bad = (cnt >= bad_neighbor_min)

                if is_bad:
                    msk[y, x] = 1

                    if mode_id == 0:  # mean: same-color cross (step=2)
                        p0 = (p2 + p4 + p5 + p7) // 4

                    elif mode_id == 1:  # gradient: pick lowest-gradient axis
                        dv  = abs(q2 - q7)
                        dh  = abs(q4 - q5)
                        ddl = abs(q1 - q8)
                        ddr = abs(q3 - q6)

                        m  = dv; idx = 0
                        if dh  < m: m, idx = dh , 1
                        if ddl < m: m, idx = ddl, 2
                        if ddr < m: m, idx = ddr, 3
                        if   idx == 0: p0 = (q2 + q7 + 1) // 2
                        elif idx == 1: p0 = (q4 + q5 + 1) // 2
                        elif idx == 2: p0 = (q1 + q8 + 1) // 2
                        else:          p0 = (q3 + q6 + 1) // 2

                    else:  # median-ish 8-neighbor estimator
                        p0 = ((min(max(q1, q8), max(q2, q7)) + min(max(q3, q6), max(q4, q5))) + 1) // 2

                if p0 < 0: p0 = 0
                if p0 > clipv: p0 = clipv
                out[y, x] = p0

        return out, msk


def dead_pixel_correction(img,
                          thres=100,
                          mode='gradient',
                          clip=4095,
                          return_mask=False,
                          bayer='RGGB',
                          bad_neighbor_min=8):
    """
    Core DPC kernel with a fixed absolute threshold.
    - Bad pixel if ≥M of 8 same-color neighbors differ by > threshold.
    - mode: {'mean','gradient','median'} replacement strategies.
    """
    h, w = img.shape
    clip = int(clip)
    mode_l = str(mode).lower()
    mode_id = 0 if mode_l == 'mean' else (1 if mode_l == 'gradient' else 2)
    bad_neighbor_min = int(min(max(bad_neighbor_min, 1), 8))
    color_lut = _bayer_to_lut(bayer)

    if '_HAS_NUMBA' in globals() and _HAS_NUMBA and (h * w >= 1_000_000):
        img_pad = np.pad(img, (2, 2), 'reflect')
        out, msk = _dpc_core_njit(img_pad, h, w, int(thres), mode_id, int(clip), color_lut, bad_neighbor_min)
        out = out.astype(img.dtype, copy=False)
        if return_mask:
            return out, msk.astype(bool)
        return out

    # Python fallback path (small images / no numba)
    img_pad = np.pad(img, (2, 2), 'reflect')
    out = np.empty_like(img)
    bad_mask = np.zeros_like(img, dtype=bool) if return_mask else None

    for y in range(h):
        py = y + 2
        for x in range(w):
            px = x + 2
            p0 = int(img_pad[py, px])

            p1 = int(img_pad[py - 2, px - 2]); p2 = int(img_pad[py - 2, px]);     p3 = int(img_pad[py - 2, px + 2])
            p4 = int(img_pad[py,     px - 2]);                                    p5 = int(img_pad[py,     px + 2])
            p6 = int(img_pad[py + 2, px - 2]); p7 = int(img_pad[py + 2, px]);     p8 = int(img_pad[py + 2, px + 2])

            cid = int(color_lut[y & 1, x & 1])  # 0=R,1=G,2=B
            if cid == 1:
                q1 = int(img_pad[py - 1, px - 1]); q2 = p2;                         q3 = int(img_pad[py - 1, px + 1])
                q4 = p4;                                                            q5 = p5
                q6 = int(img_pad[py + 1, px - 1]); q7 = p7;                         q8 = int(img_pad[py + 1, px + 1])
            else:
                q1, q2, q3, q4, q5, q6, q7, q8 = p1, p2, p3, p4, p5, p6, p7, p8

            cnt = 0
            if abs(q1 - p0) > thres: cnt += 1
            if abs(q2 - p0) > thres: cnt += 1
            if abs(q3 - p0) > thres: cnt += 1
            if abs(q4 - p0) > thres: cnt += 1
            if abs(q5 - p0) > thres: cnt += 1
            if abs(q6 - p0) > thres: cnt += 1
            if abs(q7 - p0) > thres: cnt += 1
            if abs(q8 - p0) > thres: cnt += 1

            is_bad = (cnt >= bad_neighbor_min)

            if is_bad:
                if bad_mask is not None:
                    bad_mask[y, x] = True

                if mode_l == 'mean':
                    p0 = (p2 + p4 + p5 + p7) // 4

                elif mode_l == 'gradient':
                    dv  = abs(q2 - q7)
                    dh  = abs(q4 - q5)
                    ddl = abs(q1 - q8)
                    ddr = abs(q3 - q6)
                    m = dv; idx = 0
                    if dh  < m: m, idx = dh , 1
                    if ddl < m: m, idx = ddl, 2
                    if ddr < m: m, idx = ddr, 3
                    if   idx == 0: p0 = (q2 + q7 + 1) // 2
                    elif idx == 1: p0 = (q4 + q5 + 1) // 2
                    elif idx == 2: p0 = (q1 + q8 + 1) // 2
                    else:          p0 = (q3 + q6 + 1) // 2
                else:
                    p0 = ((min(max(q1, q8), max(q2, q7)) + min(max(q3, q6), max(q4, q5))) + 1) // 2

            if p0 < 0: p0 = 0
            if p0 > clip: p0 = clip
            out[y, x] = p0

    if return_mask:
        return out, bad_mask
    return out


# ============================ I/O wrapper: single RAW ============================

def run_dpc_on_single_raw(
    raw, meta, out_dir,
    *,
    modes=('mean','gradient','median'),
    thres=100,
    thres_mode='auto',         # 'abs' / 'rel' / 'auto'
    auto_k=6.0,                # k in auto mode
    bad_neighbor_min=8,        # M for “≥M/8 outliers”
    num_bad_pixels=0,          # kept for compatibility (unused)
    pick_for_blc='gradient'
):
    """
    Production DPC wrapper: saves two previews (input/output) and returns (corrected, mask).
    New controls:
      - thres_mode: 'abs' | 'rel' | 'auto' (default 'auto')
      - auto_k: multiplier for σ in auto mode (default 6.0)
      - bad_neighbor_min: M in the “≥M/8 outliers” rule (1..8)
    """
    assert isinstance(raw, np.ndarray) and raw.ndim == 2, \
        "run_dpc_on_single_raw expects a 2D Bayer RAW numpy array"
    os.makedirs(out_dir, exist_ok=True)

    wl   = int(meta.get('white_level', 4095)) if meta is not None else 4095
    stem = os.path.splitext(meta.get('file_basename', 'raw'))[0]
    modes = tuple(modes) if isinstance(modes, (list, tuple)) else (str(modes),)

    # 1) Save input preview
    _save_gray_png_preview(os.path.join(out_dir, f"{stem}_DPC_input.png"), raw, wl, meta=meta)

    # 2) Resolve absolute threshold
    bayer = str(meta.get('bayer', 'RGGB') or 'RGGB').upper()
    m = (str(thres_mode) if thres_mode is not None else 'auto').lower()
    th_abs = None
    if m == 'abs':
        th_abs = int(max(1, int(thres)))
        src = f"abs({th_abs})"
    elif m == 'rel':
        rel = float(thres)
        # Interpret as a WL ratio (e.g., 0.006 = 0.6% of WL).
        th_abs = int(max(1, round(rel * wl)))
        src = f"rel({rel:.6f})*WL={th_abs}"
    else:
        # auto: estimate σ then multiply by k
        sigma = _estimate_sigma_samecolor(raw, bayer=bayer)
        th_abs = int(max(1, round(float(auto_k) * float(sigma))))
        # Clip into a safe band to avoid extremely small/large thresholds.
        lo = int(max(1, round(0.001 * wl)))   # 0.1% WL
        hi = int(max(lo, round(0.05 * wl)))   # 5% WL
        th_abs = int(min(max(th_abs, lo), hi))
        src = f"auto(k={float(auto_k):.3f}, sigma≈{sigma:.2f})→{th_abs}  (clipped to [{lo},{hi}])"
    logger.info("[DPC] threshold source: %s", src)
    logger.info("[DPC] bad_neighbor_min=%d  mode_pick=%s", int(bad_neighbor_min), str(pick_for_blc))

    # 3) Pick a mode and run
    pick = str(pick_for_blc).lower()
    if pick not in [str(m).lower() for m in modes]:
        logger.warning("[DPC] pick_for_blc=%s not in modes=%s; fallback to the first", pick_for_blc, modes)
        pick = str(modes[0]).lower()

    corrected, mask = dead_pixel_correction(
        raw.copy(),
        thres=int(th_abs),
        mode=pick,
        clip=wl,
        return_mask=True,
        bayer=bayer,
        bad_neighbor_min=int(bad_neighbor_min)
    )

    # 4) Save output preview
    _save_gray_png_preview(os.path.join(out_dir, f"{stem}_DPC_output_{pick}.png"), corrected, wl, meta=meta)

    logger.info("[DPC] done. mode=%s thres=%d (src=%s) bayer=%s saved=%s / %s",
                pick, int(th_abs), str(m), bayer,
                f"{stem}_DPC_input.png", f"{stem}_DPC_output_{pick}.png")
    return corrected, mask
