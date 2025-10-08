# -*- coding: utf-8 -*-
"""
BLC (Black Level Correction) — production auto-tuning.
Goal: automatically estimate per-plane black levels (R/Gr/Gb/B) for Bayer RAW
and shift the baseline so blacks sit near 0 without over-subtraction.

Two-stage search:
1) Build candidate BLs (visible low-quantile × scales; optional: meta BL × scales)
   -> score quickly on a dark ROI + sampling, keep top-K.
2) Re-check top-K on the full image -> drop high-negative candidates -> pick best.
   -> Apply per-plane subtraction; produce [0,1] preview for analysis and linear x_lin for downstream.

Robustness & speed:
- Use a dark ROI to avoid bright/saturated bias; grid/random sampling for speed.
- Candidates from quantiles × scales reduce under/over subtraction; stronger penalty on negatives.
- Per-plane subtraction/division for memory efficiency; clamp BL to keep denominators safe.
"""

import os, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# ---------- Bayer layout ----------
# Map the four Bayer phases to (row_offset, col_offset).
_BAYER_OFFSETS = {
    "RGGB": ((0,0), (0,1), (1,0), (1,1)),  # R, Gr, Gb, B
    "GRBG": ((0,1), (0,0), (1,1), (1,0)),
    "GBRG": ((1,0), (1,1), (0,0), (0,1)),
    "BGGR": ((1,1), (1,0), (0,1), (0,0)),
}

def _norm_bayer(pat: str) -> str:
    # Normalize Bayer string; raise on unknown patterns.
    pat = (pat or "RGGB").upper()
    if pat not in _BAYER_OFFSETS:
        raise ValueError(f"Unsupported bayer pattern: {pat}")
    return pat

# ---------- Utilities ----------
def _low_percentile_mean(x, p=0.01):
    """
    Mean of the darkest p-quantile.
    First drop the top 95% tail to suppress bright/hot pixels, then average the lowest p.
    """
    x = np.asarray(x, np.float32).ravel()
    if x.size == 0:
        return 0.0
    hi = np.percentile(x, 95.0)  # suppress bright tail
    x = x[x <= hi]
    k = max(1, int(len(x) * float(p)))
    part = np.partition(x, k - 1)[:k]
    return float(np.mean(part))

def _estimate_four_channel_bl(raw, bayer="RGGB", p=0.01, exclude_mask=None):
    """
    Estimate per-plane BL (R/Gr/Gb/B) via low-quantile means.
    Pixels masked by exclude_mask are skipped (e.g., bad/hot pixels).
    """
    pat = _norm_bayer(bayer)
    (yR,xR),(yGr,xGr),(yGb,xGb),(yB,xB) = _BAYER_OFFSETS[pat]

    def ch_stat(ch_y, ch_x):
        arr = raw[ch_y::2, ch_x::2]
        if exclude_mask is not None:
            m = np.asarray(exclude_mask, dtype=bool)[ch_y::2, ch_x::2]
            arr = arr[~m]
        return _low_percentile_mean(arr, p)

    R  = ch_stat(yR,  xR)
    Gr = ch_stat(yGr, xGr)
    Gb = ch_stat(yGb, xGb)
    B  = ch_stat(yB,  xB)
    return {"R": float(R), "Gr": float(Gr), "Gb": float(Gb), "B": float(B)}

def _to_bl4(v):
    """
    Normalize external BL inputs into a dict:
    scalar -> duplicate to all planes; list(4) -> map to R/Gr/Gb/B; dict -> fill missing with 0.
    """
    if v is None:
        return None
    if np.isscalar(v):
        f = float(v); return {"R":f,"Gr":f,"Gb":f,"B":f}
    if isinstance(v, dict):
        return {"R":float(v.get("R",0)), "Gr":float(v.get("Gr",0)),
                "Gb":float(v.get("Gb",0)), "B":float(v.get("B",0))}
    v = list(map(float, v))
    if len(v) == 4:
        return {"R":v[0], "Gr":v[1], "Gb":v[2], "B":v[3]}
    f = float(v[0]); return {"R":f,"Gr":f,"Gb":f,"B":f}

def _infer_white_level(meta, raw) -> float:
    """
    Infer WL priority:
    1) meta.white_level
    2) meta.bit_depth -> (1<<bit)-1
    3) dtype limit for integer RAW
    4) default to 12-bit
    """
    WL = meta.get("white_level", None)
    if WL is not None:
        return float(WL)
    bd = meta.get("bit_depth", None)
    if bd is not None:
        try:
            bd = int(bd)
            return float((1 << bd) - 1)
        except Exception:
            pass
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        return float(info.max)
    return float((1 << 12) - 1)

def _clip_bl4_to_wl(bl4, WL, eps=1.0):
    """
    Clamp BL to [0, WL-eps] to keep (WL - BL) sufficiently large.
    Returns the clamped dict and whether clamping happened.
    """
    bl_safe = {k: float(max(0.0, min(float(v), float(WL) - float(eps))))
               for k, v in bl4.items()}
    clipped = any(bl_safe[k] != float(bl4.get(k, 0.0)) for k in ("R","Gr","Gb","B"))
    return bl_safe, clipped

# ---------- Apply BLC (memory-friendly) ----------
def _apply_blc_basic(raw, bayer, WL, bl4):
    """
    Per-plane subtraction and normalization:
    - Subtract BL per plane.
    - Normalize by (WL - BL_c) per plane -> out01 in [0,1] (for inspection only).
    Returns out01, negative fraction, and x_lin (subtracted linear RAW).
    """
    raw = np.asarray(raw)
    assert raw.ndim == 2, "raw must be HxW"
    pat = _norm_bayer(bayer)
    (yR,xR),(yGr,xGr),(yGb,xGb),(yB,xB) = _BAYER_OFFSETS[pat]

    x = raw.astype(np.float32, copy=True)
    bl4 = {k: float(bl4.get(k, 0.0)) for k in ("R","Gr","Gb","B")}
    bl4, _ = _clip_bl4_to_wl(bl4, WL, eps=1.0)

    # Subtract per plane in place
    x[yR::2,  xR::2]  -= bl4["R"]
    x[yGr::2, xGr::2] -= bl4["Gr"]
    x[yGb::2, xGb::2] -= bl4["Gb"]
    x[yB::2,  xB::2]  -= bl4["B"]

    neg_frac = float((x < 0).mean())  # over-subtraction indicator

    # Normalize per plane with scalars: out01 = (raw - BL_c)/(WL - BL_c)
    y01 = np.empty_like(x, dtype=np.float32)
    dR  = max(float(WL) - bl4["R"],  1.0)
    dGr = max(float(WL) - bl4["Gr"], 1.0)
    dGb = max(float(WL) - bl4["Gb"], 1.0)
    dB  = max(float(WL) - bl4["B"],  1.0)

    y01[yR::2,  xR::2]  = x[yR::2,  xR::2]  / dR
    y01[yGr::2, xGr::2] = x[yGr::2, xGr::2] / dGr
    y01[yGb::2, xGb::2] = x[yGb::2, xGb::2] / dGb
    y01[yB::2,  xB::2]  = x[yB::2,  xB::2]  / dB

    y01 = np.clip(y01, 0.0, 1.0)
    return y01, neg_frac, x

# ---------- Visualization / metrics ----------
def _to_u8(img01, black_q=0.01, white_q=0.999):
    # Stretch [0,1] image to 8-bit using shared quantile mapping for visual comparison.
    x = np.asarray(img01, np.float32)
    lo = float(np.quantile(x, black_q)) if x.size else 0.0
    hi = float(np.quantile(x, white_q)) if x.size else 1.0
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x*255.0 + 0.5).astype(np.uint8)

def _build_eval_mask(inp01, *, use_dark_roi=True, roi_quantile=0.20,
                     stride=2, fraction=None, seed=42, sat_clip=0.98):
    """
    Build an evaluation mask:
    - Dark ROI: keep pixels <= q(roi_quantile); drop near-saturated samples.
    - Sampling: grid (stride) and/or random fraction for speed; can be combined.
    - Fallback: all True if mask becomes empty.
    """
    h, w = inp01.shape
    mask = np.ones((h, w), dtype=bool)

    if use_dark_roi:
        thr = float(np.quantile(inp01, roi_quantile)) if inp01.size else 0.0
        mask &= (inp01 <= thr)
    mask &= (inp01 < float(sat_clip))

    if stride is not None and int(stride) > 1:
        s = int(stride)
        grid = np.zeros_like(mask)
        grid[::s, ::s] = True          # covers all Bayer phases
        mask &= grid

    if fraction is not None and 0.0 < float(fraction) < 1.0:
        rng = np.random.default_rng(seed)
        pick = rng.random(mask.shape) < float(fraction)
        mask &= pick

    if not mask.any():
        mask[:] = True
    return mask

def _make_metrics(meta, inp01, out01, neg_frac_out, bl_used, roi_mask=None):
    """
    Compute metrics and a score (heavier penalty on negatives):
    - Peak shift toward 0 indicates better BL.
    - Low quantiles improvement matters.
    - Negative fraction penalizes over-subtraction.
    Score: 0.45*peak + 0.25*q + 0.30*(1 - neg_penalty).
    """
    WL = float(meta.get("white_level"))

    # Evaluate on ROI when available
    if roi_mask is not None and roi_mask.shape == inp01.shape and roi_mask.any():
        inp_eval = np.asarray(inp01, np.float32)[roi_mask]
        out_eval = np.asarray(out01, np.float32)[roi_mask]
    else:
        inp_eval = np.asarray(inp01, np.float32).ravel()
        out_eval = np.asarray(out01, np.float32).ravel()

    def _hist_peak_1d(x01, bins=4096):
        hist, bins = np.histogram(x01, bins=bins, range=(0,1))
        i = int(np.argmax(hist))
        return float(0.5*(bins[i]+bins[i+1]))

    def _q_1d(x, p):
        return float(np.quantile(x, float(p)))

    m = {
        "file": meta.get("file_basename"),
        "bayer": meta.get("bayer"),
        "white_level": WL,
        "black_level_used": {k: float(v) for k, v in bl_used.items()},
        "mean_std": {
            "input":  [float(inp_eval.mean()),  float(inp_eval.std())],
            "output": [float(out_eval.mean()), float(out_eval.std())],
        },
        "hist_peak": {
            "input":  _hist_peak_1d(inp_eval),
            "output": _hist_peak_1d(out_eval),
        },
        "low_quantiles": {
            "input":  {"q001": _q_1d(inp_eval,0.01), "q005": _q_1d(inp_eval,0.05)},
            "output": {"q001": _q_1d(out_eval,0.01), "q005": _q_1d(out_eval,0.05)},
        },
        "neg_fraction": {
            "input":  float((inp_eval < 0).mean() if inp_eval.size else 0.0),
            "output": float(neg_frac_out),
        }
    }

    # ---- Scoring (stronger negative penalty to avoid left-bias) ----
    hp_in, hp_out = m["hist_peak"]["input"], m["hist_peak"]["output"]
    q1_in, q1_out = m["low_quantiles"]["input"]["q001"], m["low_quantiles"]["output"]["q001"]
    neg_out = m["neg_fraction"]["output"]

    delta_peak = max(hp_in - hp_out, 0.0)                 # shift toward 0 (larger is better)
    norm_peak  = min(1.0, delta_peak / 0.005)
    base       = max(abs(q1_in), 1e-6)
    norm_q     = min(1.0, max(q1_in - q1_out, 0.0) / base)
    neg_penalty= min(1.0, neg_out / 0.01)                 # 1% negatives => full penalty

    w_peak, w_q, w_neg = 0.45, 0.25, 0.30
    score = w_peak * norm_peak + w_q * norm_q + w_neg * (1.0 - neg_penalty)

    m["bl_score"] = {
        "score": float(np.clip(score, 0.0, 1.0)),
        "norm_peak": norm_peak, "norm_q001": norm_q,
        "neg_penalty": neg_penalty, "delta_peak": delta_peak
    }
    return m

# ---------- Candidate eval / auto search ----------
def _eval_candidate(raw, meta, inp01, base_bl, scale, roi_mask=None):
    """
    Evaluate a candidate BL (= base_bl × scale):
    Apply BLC, compute metrics/score on ROI or full image, and return metrics and scaled BL.
    """
    bl_scaled = {k: float(v) * float(scale) for k, v in base_bl.items()}
    out01, neg_frac_global, x_lin = _apply_blc_basic(raw, meta["bayer"], meta["white_level"], bl_scaled)

    if roi_mask is not None and roi_mask.shape == raw.shape and roi_mask.any():
        neg_frac_roi = float((x_lin[roi_mask] < 0).mean())
        metrics = _make_metrics(meta, inp01, out01, neg_frac_roi, bl_scaled, roi_mask=roi_mask)
    else:
        metrics = _make_metrics(meta, inp01, out01, neg_frac_global, bl_scaled, roi_mask=None)

    return metrics, out01, bl_scaled

def _autosweep_best(raw, meta, inp01, p_list, scale_list,
                    neg_max=0.002, eval_mask=None, topk_full=3, exclude_mask=None):
    """
    Two-stage auto sweep:
    Stage 1 (fast): score all candidates on ROI/samples -> keep top-K.
    Stage 2 (robust): re-score top-K on full image + filter by negative ratio -> pick best.
    Candidates are from: (optional) meta BL × scales and visible low-quantile BL × scales.
    """
    # ---------- Stage 1: ROI/scored sampling ----------
    stage1 = []

    # 1) meta BL (if present)
    meta_bl = _to_bl4(meta.get("black_level", None))
    if meta_bl is not None:
        for s in scale_list:
            m, out, bl = _eval_candidate(raw, meta, inp01, meta_bl, s, roi_mask=eval_mask)
            stage1.append(("meta", None, s, m, bl))

    # 2) low-quantile estimate × scale (supports exclude_mask)
    for p in p_list:
        base_bl = _estimate_four_channel_bl(raw, meta["bayer"], p=float(p),
                                            exclude_mask=exclude_mask)
        for s in scale_list:
            m, out, bl = _eval_candidate(raw, meta, inp01, base_bl, s, roi_mask=eval_mask)
            stage1.append(("visible", float(p), s, m, bl))

    # Keep top-K by ROI score
    stage1.sort(key=lambda t: -t[3]["bl_score"]["score"])
    shortlist = stage1[:max(1, int(topk_full))]

    # ---------- Stage 2: full-image verification ----------
    final_pool = []
    for origin, p, s, m_roi, bl in shortlist:
        m_full, out_full, bl_used = _eval_candidate(raw, meta, inp01, bl, 1.0, roi_mask=None)
        rec = {
            "origin": origin,
            "estimate_p": p,
            "scale": float(s),
            "score": float(m_full["bl_score"]["score"]),
            "neg_out": float(m_full["neg_fraction"]["output"]),
            "q001_out": float(m_full["low_quantiles"]["output"]["q001"]),
            "q005_out": float(m_full["low_quantiles"]["output"]["q005"]),
            "hist_peak_out": float(m_full["hist_peak"]["output"]),
            "bl_used": bl_used,
            "metrics_full": m_full,
            "metrics_roi": m_roi
        }
        final_pool.append(rec)

    # Filter by negative ratio; if empty, fall back to unfiltered pool.
    valid = [r for r in final_pool if r["neg_out"] <= float(neg_max)]
    pool = valid if valid else final_pool
    # Sort by score, then q001, negatives, and distance of scale to 1.0 (more conservative).
    pool.sort(key=lambda r: (-r["score"], r["q001_out"], r["neg_out"], abs(r["scale"]-1.0)))
    best = pool[0]
    return best, final_pool

# ---------- Visualization (histogram / low-tone CDF) ----------
def _plot_hist(inp01, out01, save_full, save_low, low_max=0.05, bins_full=4096, bins_low=2048):
    plt.figure(figsize=(8,5))
    plt.hist(inp01.ravel(),  bins=bins_full, range=(0,1), alpha=0.5, label="input",  density=True)
    plt.hist(out01.ravel(), bins=bins_full, range=(0,1), alpha=0.5, label="output (BL only)", density=True)
    plt.legend(); plt.xlabel("Intensity (0..1)"); plt.ylabel("PDF"); plt.title("Histogram (full)")
    plt.tight_layout(); plt.savefig(save_full, dpi=120); plt.close()

    plt.figure(figsize=(8,5))
    plt.hist(inp01.ravel(),  bins=bins_low, range=(0,low_max), alpha=0.5, label="input",  density=True)
    plt.hist(out01.ravel(), bins=bins_low, range=(0,low_max), alpha=0.5, label="output (BL only)", density=True)
    plt.legend(); plt.xlabel(f"Intensity (0..{low_max})"); plt.ylabel("PDF"); plt.title("Histogram (low tones)")
    plt.tight_layout(); plt.savefig(save_low, dpi=120); plt.close()

def _plot_cdf(inp01, out01, save_path, low_max=0.05, bins=4000):
    hist_in,  bin_edges = np.histogram(inp01.ravel(),  bins=bins, range=(0, low_max), density=True)
    hist_out, _        = np.histogram(out01.ravel(), bins=bins, range=(0, low_max), density=True)
    cdf_in  = np.cumsum(hist_in)  / max(np.sum(hist_in), 1e-12)
    cdf_out = np.cumsum(hist_out) / max(np.sum(hist_out), 1e-12)
    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    plt.figure(figsize=(8,5))
    plt.plot(x, cdf_in,  label="input")
    plt.plot(x, cdf_out, label="output (BL only)")
    plt.xlabel(f"Intensity (0..{low_max})"); plt.ylabel("CDF"); plt.title("Low-tone CDF")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()

# ---------- Public entry ----------
def _ensure_float_list(v, default):
    """
    Normalize strings/arrays/None into a float list.
    Examples: None -> default; "0.95,1.0" -> [0.95,1.0]; [0.95,1.0] -> same.
    """
    if v is None:
        return list(default)
    if isinstance(v, str):
        toks = [t.strip() for t in v.split(",") if t.strip()]
        return [float(t) for t in toks] if toks else list(default)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    # Fallback to default.
    return list(default)

def run_auto_blc(raw, meta, out_dir,
                 p_list=None, scale_list=None, neg_max=0.002,
                 save_plots=True,
                 return_mode="legacy",
                 exclude_mask=None,
                 # Speed/robust defaults
                 use_dark_roi=True, roi_quantile=0.20,
                 eval_stride=2, eval_fraction=None, topk_full=3):
    """
    Production auto BLC.
    Inputs: candidate quantiles (p_list), scales (scale_list), negative cap, optional mask.
    Returns by mode:
    - "legacy": (out01_best, best_record, metrics)
    - "lin":    (x_lin_best, meta_out, info)
    - "both":   both above; info['out01'] also attached
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(meta)
    assert raw.ndim == 2, "raw must be HxW"
    meta["bayer"] = _norm_bayer(meta.get("bayer", "RGGB"))
    WL = _infer_white_level(meta, raw)
    meta["white_level"] = float(WL)
    meta.setdefault("file_basename", "image.dng")

    # Input mapped to [0,1] for analysis/plots only
    inp01 = np.clip(raw.astype(np.float32) / max(WL, 1e-6), 0, 1)

    # Parse candidate lists
    p_list = _ensure_float_list(p_list,  [0.008, 0.010, 0.012, 0.015, 0.020])
    scale_list = _ensure_float_list(scale_list, [0.95, 0.98, 1.00, 1.02, 1.05, 1.08])

    # Stage-1 evaluation mask (dark ROI + sampling)
    eval_mask = _build_eval_mask(inp01, use_dark_roi=use_dark_roi,
                                 roi_quantile=roi_quantile,
                                 stride=eval_stride, fraction=eval_fraction)

    # Two-stage search
    best, table = _autosweep_best(raw, meta, inp01,
                                  p_list, scale_list,
                                  neg_max=neg_max,
                                  eval_mask=eval_mask,
                                  topk_full=topk_full,
                                  exclude_mask=exclude_mask)

    # Apply best BL -> out01 (for plots) + x_lin (downstream)
    out01_best, neg_frac_best, x_lin_best = _apply_blc_basic(raw, meta["bayer"], WL, best["bl_used"])
    metrics = _make_metrics(meta, inp01, out01_best, neg_frac_best, best["bl_used"])
    metrics["auto_params"] = {"estimate_p": best.get("estimate_p"), "scale": best["scale"], "origin": best["origin"]}

    # Optional plots
    if save_plots:
        stem = os.path.splitext(os.path.basename(meta.get("file_basename", "image")))[0]
        tag  = f"auto_p{(best.get('estimate_p') if best.get('estimate_p') is not None else -1):.3f}_s{best['scale']:.3f}"
        try:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_input_8bit.png"), _to_u8(inp01))
            cv2.imwrite(os.path.join(out_dir, f"{stem}_blc_output_8bit.png"), _to_u8(out01_best))
            side = np.hstack([_to_u8(inp01), _to_u8(out01_best)])
            cv2.imwrite(os.path.join(out_dir, f"{stem}_side_by_side_{tag}.png"), side)
        except Exception as e:
            logger.warning("[BLC] plot save failed: %s", e)

    logger.info("[BLC/auto] best origin=%s  p=%s  scale=%.3f  score=%.3f  neg=%.6f",
                best['origin'], str(best.get('estimate_p')), best['scale'],
                metrics['bl_score']['score'], metrics['neg_fraction']['output'])

    # === Return by mode ===
    if return_mode == "legacy":
        return out01_best, best, metrics

    meta_out = dict(meta)
    meta_out["black_level_used"] = {k: float(v) for k, v in best["bl_used"].items()}

    info = {
        "bl_used": best["bl_used"],
        "best_record": best,   # stage-1/2 details
        "metrics": metrics     # final metrics & score
    }
    if return_mode == "both":
        info["out01"] = out01_best

    if return_mode in ("lin", "both"):
        return x_lin_best, meta_out, info

    raise ValueError("return_mode must be one of: 'legacy', 'lin', 'both'")
