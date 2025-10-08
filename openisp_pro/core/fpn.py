# core/fpn.py
# -*- coding: utf-8 -*-
import os
import json
import numpy as np

def _bayer_offsets(bayer):
    bayer = (bayer or "RGGB").upper()
    mp = {
        "RGGB": ((0,0),(0,1),(1,0),(1,1)),
        "GRBG": ((0,1),(0,0),(1,1),(1,0)),
        "GBRG": ((1,0),(1,1),(0,0),(0,1)),
        "BGGR": ((1,1),(1,0),(0,1),(0,0)),
    }
    return mp[bayer]

def _rolling_median_1d(x, radius):
    """1D rolling median smoothing; small radius (e.g., 7–31) tames jitter."""
    n = len(x)
    if radius <= 0: return x
    y = np.empty_like(x)
    for i in range(n):
        s = max(0, i - radius); e = min(n, i + radius + 1)
        y[i] = np.median(x[s:e])
    return y

def _robust_center(x, pct=(0.1, 0.9)):
    """Median after quantile clipping to reduce outlier influence."""
    lo, hi = np.quantile(x, pct)
    x2 = x[(x >= lo) & (x <= hi)]
    if x2.size == 0: return float(np.median(x))
    return float(np.median(x2))

def _gradient_mask(raw01, k=4.0):
    """Simple gradient weight: downweight edges to avoid biasing row/col stats."""
    gy = np.abs(np.diff(raw01, axis=0, prepend=raw01[[0], :]))
    gx = np.abs(np.diff(raw01, axis=1, prepend=raw01[:, [0]]))
    g = gx + gy
    med = np.median(g)
    if med <= 1e-8: med = 1e-8
    w = 1.0 / (1.0 + (g / (k * med))**2)
    return w

def _channel_view(raw, off):
    y, x = off
    return raw[y::2, x::2]

def _apply_channel_view(raw, off, patch):
    y, x = off
    raw[y::2, x::2] -= patch

def _estimate_row_bias(ch, w=None, clip_pct=(0.05, 0.95)):
    """Per-row median (with weights/clipping), then subtract global center → row bias."""
    H, W = ch.shape
    if w is None: w = np.ones_like(ch, dtype=np.float32)
    row_med = np.zeros(H, dtype=np.float32)
    for r in range(H):
        v = ch[r, :]
        ww = w[r, :]
        if ww is not None:
            # Use pixels with weight > 0.5 * mean weight
            thr = 0.5 * float(np.mean(ww))
            m = ww > thr
        else:
            m = np.ones_like(v, dtype=bool)
        v2 = v[m]
        if v2.size == 0:
            row_med[r] = np.median(v)
        else:
            lo, hi = np.quantile(v2, clip_pct)
            v3 = v2[(v2 >= lo) & (v2 <= hi)]
            row_med[r] = np.median(v3) if v3.size else np.median(v2)
    global_c = _robust_center(row_med, (0.1, 0.9))
    bias = row_med - global_c
    return bias

def _estimate_col_bias(ch, w=None, clip_pct=(0.05, 0.95)):
    """Per-column median (with weights/clipping), then subtract global center → col bias."""
    H, W = ch.shape
    if w is None: w = np.ones_like(ch, dtype=np.float32)
    col_med = np.zeros(W, dtype=np.float32)
    for c in range(W):
        v = ch[:, c]
        ww = w[:, c]
        if ww is not None:
            thr = 0.5 * float(np.mean(ww))
            m = ww > thr
        else:
            m = np.ones_like(v, dtype=bool)
        v2 = v[m]
        if v2.size == 0:
            col_med[c] = np.median(v)
        else:
            lo, hi = np.quantile(v2, clip_pct)
            v3 = v2[(v2 >= lo) & (v2 <= hi)]
            col_med[c] = np.median(v3) if v3.size else np.median(v2)
    global_c = _robust_center(col_med, (0.1, 0.9))
    bias = col_med - global_c
    return bias

def run_fpn_suppress(raw_lin, meta, out_dir,
                     mode="rowcol", per_channel=True,
                     smooth_radius=9, strength=1.0,
                     exclude_mask=None, save_plots=True):
    """
    Lightweight row/column FPN suppression (additive bias).
    Recommended after BLC and before BM3D.
      - mode: 'row' / 'col' / 'rowcol'
      - per_channel: estimate on each Bayer subchannel
      - smooth_radius: rolling-median radius for bias curves
      - strength: 0..1 scale of subtraction
      - exclude_mask: True to exclude pixels (e.g., hot/saturated)
    """
    os.makedirs(out_dir, exist_ok=True)
    bayer = meta.get("bayer", "RGGB")
    WL = float(meta.get("white_level", 4095.0))
    x = raw_lin.astype(np.float32).copy()

    # Normalize for weights/preview only; subtraction remains in original scale.
    x01 = np.clip(x / max(WL, 1e-6), 0.0, 1.0)
    w_grad = _gradient_mask(x01, k=4.0)

    if exclude_mask is not None:
        # Excluded pixels get low weight
        w = w_grad * (~exclude_mask).astype(np.float32)
    else:
        w = w_grad

    offs = _bayer_offsets(bayer)
    # Per-channel views
    chs = [_channel_view(x, off) for off in offs]
    ws  = [_channel_view(w, off) for off in offs]

    info = {"mode": mode, "per_channel": per_channel, "smooth_radius": smooth_radius,
            "strength": float(strength)}

    def _apply_row(ch, wch, off):
        bias_r = _estimate_row_bias(ch, wch)
        if smooth_radius and smooth_radius > 0:
            bias_r = _rolling_median_1d(bias_r, smooth_radius)
        patch = bias_r[:, None] * strength
        _apply_channel_view(x, off, patch)
        return bias_r

    def _apply_col(ch, wch, off):
        bias_c = _estimate_col_bias(ch, wch)
        if smooth_radius and smooth_radius > 0:
            bias_c = _rolling_median_1d(bias_c, smooth_radius)
        patch = bias_c[None, :] * strength
        _apply_channel_view(x, off, patch)
        return bias_c

    # Stripe metric before: std of row/col bias (diagnostic only)
    def _stripe_metric(ch):
        rb = _estimate_row_bias(ch, None)
        cb = _estimate_col_bias(ch, None)
        return float(np.std(rb)), float(np.std(cb))

    metrics_before = []
    for ch in chs:
        metrics_before.append(_stripe_metric(ch))

    # Apply corrections
    row_biases = []; col_biases = []
    if mode in ("row", "rowcol"):
        for ch, wch, off in zip(chs, ws, offs):
            row_biases.append(_apply_row(ch, wch, off))
    if mode in ("col", "rowcol"):
        # Refresh views after row correction
        chs = [_channel_view(x, off) for off in offs]
        ws  = [_channel_view(w, off) for off in offs]
        for ch, wch, off in zip(chs, ws, offs):
            col_biases.append(_apply_col(ch, wch, off))

    # Metrics after
    chs_after = [_channel_view(x, off) for off in offs]
    metrics_after = []
    for ch in chs_after:
        metrics_after.append(_stripe_metric(ch))

    # Minimal visualization (optional)
    if save_plots:
        try:
            import imageio.v2 as imageio
            import matplotlib.pyplot as plt
            def to8(u):
                u01 = np.clip(u / max(WL, 1e-6), 0, 1)
                return (u01 * 255.0 + 0.5).astype(np.uint8)
            imageio.imwrite(os.path.join(out_dir, "fpn_input_preview.png"), to8(raw_lin))
            imageio.imwrite(os.path.join(out_dir, "fpn_output_preview.png"), to8(x))
            # Plot first channel’s curves only to keep output small
            if row_biases:
                rb0 = row_biases[0]
                plt.figure(); plt.plot(rb0); plt.title("Row bias (R ch)"); plt.savefig(os.path.join(out_dir,"row_bias_R.png")); plt.close()
            if col_biases:
                cb0 = col_biases[0]
                plt.figure(); plt.plot(cb0); plt.title("Col bias (R ch)"); plt.savefig(os.path.join(out_dir,"col_bias_R.png")); plt.close()
        except Exception:
            pass

    info["metrics_before"] = [{"row_std": r, "col_std": c} for (r,c) in metrics_before]
    info["metrics_after"]  = [{"row_std": r, "col_std": c} for (r,c) in metrics_after]

    meta_out = dict(meta)
    return x, meta_out, info
