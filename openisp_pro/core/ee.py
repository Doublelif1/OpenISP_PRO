# core/ee.py
# -*- coding: utf-8 -*-
"""
Y' Edge Enhancement after CSC/TMC (YUV444)
- Enhance Y' only; pass U/V through to avoid chroma shifts.
- USM with noise gating, edge mask, tone protection, and asymmetric halo control.
- Accepts uint8/uint16/float; normalizes to [0,1] float internally; outputs uint8 (aligned with TMC).
- 8-bit friendly gating: supports LSB threshold + MAD σ, clamped to a given LSB band so effects remain visible.
"""

from __future__ import annotations
import os, json
import numpy as np
from typing import Dict, Tuple
from PIL import Image
import logging
logger = logging.getLogger("yuv_edge_enhance")

_EPS = 1e-8


# ---------- Utilities ----------

def _to_float01(x: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] from u8/u16/float; clip float inputs to [0,1]."""
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    if np.issubdtype(x.dtype, np.floating):
        return np.clip(x.astype(np.float32), 0.0, 1.0)
    if np.issubdtype(x.dtype, np.integer):
        mx = float(np.iinfo(x.dtype).max)
        return np.clip(x.astype(np.float32) / max(mx, 1.0), 0.0, 1.0)
    return np.clip(x.astype(np.float32), 0.0, 1.0)

def _sanitize01(x: np.ndarray) -> np.ndarray:
    """Normalize and replace NaN/Inf with safe defaults."""
    y = _to_float01(x)
    return np.nan_to_num(y, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

def _to_uint8(x01: np.ndarray) -> np.ndarray:
    """Quantize [0,1] float to uint8 with rounding."""
    return (np.clip(x01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def _to_uint8_vis(x: np.ndarray) -> np.ndarray:
    """Auto-window to [1%,99%] for visualization, then convert to uint8."""
    x = np.asarray(x, np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return _to_uint8(np.zeros_like(x))
    lo, hi = np.percentile(x[finite], [1.0, 99.0])
    if hi <= lo + 1e-12:
        return _to_uint8(np.zeros_like(x))
    return _to_uint8((x - lo) / (hi - lo))

def _save_gray_png(path: str, x01_or_any: np.ndarray, assume01: bool = True) -> None:
    """Save a grayscale PNG; assume already [0,1] or auto-window for visibility."""
    try:
        arr = _to_uint8(x01_or_any) if assume01 else _to_uint8_vis(x01_or_any)
        Image.fromarray(arr).save(path)
    except Exception as e:
        logger.warning("[EE] 保存灰度图失败 %s: %s", path, e)

def _box_filter_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Fast box blur via summed-area table; radius<=0 returns input."""
    if radius <= 0:
        return np.asarray(x, np.float32, copy=False)
    x = np.asarray(x, np.float32)
    pad = int(radius)
    xp = np.pad(x, ((pad, pad), (pad, pad)), mode='edge').astype(np.float64, copy=False)
    I = np.cumsum(np.cumsum(np.pad(xp, ((1, 0), (1, 0)), mode='constant'), axis=0), axis=1, dtype=np.float64)
    k = 2 * pad + 1
    out = I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]
    out = out / float(k * k)
    return out.astype(np.float32, copy=False)

def _blur_gauss_box(x: np.ndarray, radius: int, passes: int = 2) -> np.ndarray:
    """Approximate Gaussian blur with repeated box filters."""
    y = np.asarray(x, np.float32)
    for _ in range(max(int(passes), 1)):
        y = _box_filter_2d(y, radius)
    return y

def _grad_energy(x: np.ndarray, preblur_radius: int = 0) -> np.ndarray:
    """Gradient magnitude squared (optional pre-blur); central differences with edge handling."""
    x = np.asarray(x, np.float32)
    if preblur_radius > 0:
        x = _box_filter_2d(x, preblur_radius)
    gx = np.zeros_like(x); gy = np.zeros_like(x)
    gx[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    gx[:, 0] = x[:, 1] - x[:, 0]
    gx[:, -1] = x[:, -1] - x[:, -2]
    gy[1:-1, :] = 0.5 * (x[2:, :] - x[:-2, :])
    gy[0, :] = x[1, :] - x[0, :]
    gy[-1, :] = x[-1, :] - x[-2, :]
    return gx * gx + gy * gy

def _smoothstep(a, b, v: np.ndarray) -> np.ndarray:
    """Smoothstep gate between a and b."""
    t = (np.asarray(v, np.float32) - np.asarray(a, np.float32)) / np.maximum(np.asarray(b, np.float32) - np.asarray(a, np.float32), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _safe_quantile(x: np.ndarray, q: float) -> float:
    """Robust quantile with finite check."""
    x = np.asarray(x, np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, float(np.clip(q, 0.0, 1.0)), method="linear"))

def _mad_sigma(x: np.ndarray) -> float:
    """MAD-based sigma estimator for gating."""
    x = np.asarray(x, np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


# ---------- Preview helpers (align keys with pipeline/TMC) ----------

def _yuv2rgb_preview(Y01, U01, V01, meta: Dict) -> np.ndarray | None:
    """Lightweight YUV444→RGB preview using provided coeffs or BT.601 full fallback."""
    try:
        rng = "full"
        if isinstance(meta, dict):
            rng = (meta.get("range") or meta.get("tmc_range") or meta.get("csc_range") or "full").lower()
        uvc = float(meta.get("uv_center", 0.5)) if isinstance(meta, dict) else 0.5

        coeffs = None
        if isinstance(meta, dict):
            coeffs = meta.get("coeffs") or meta.get("csc_coeffs")

        y = Y01.astype(np.float32); u = U01.astype(np.float32); v = V01.astype(np.float32)
        if rng == "limited":
            y = (y - 16.0/255.0) / (219.0/255.0)
            u = (u - 16.0/255.0) / (224.0/255.0)
            v = (v - 16.0/255.0) / (224.0/255.0)

        if isinstance(coeffs, dict) and ("M_yuv2rgb" in coeffs):
            M = np.asarray(coeffs["M_yuv2rgb"], np.float32).reshape(3,3)
            off = np.asarray(coeffs.get("off_yuv2rgb", [0,0,0]), np.float32).reshape(1,1,3)
            YUV = np.stack([y, u, v], axis=-1)
            rgb = YUV @ M.T + off
        else:
            # BT.601 full-range fallback
            u0 = u - uvc; v0 = v - uvc
            r = y + 1.402 * v0
            g = y - 0.344136 * u0 - 0.714136 * v0
            b = y + 1.772 * u0
            rgb = np.stack([r,g,b], axis=-1)
        return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    except Exception as e:
        logger.warning("[EE] 预览 RGB 失败：%s", e)
        return None


# ---------- Main pipeline ----------

def run_edge_enhance_after_tmc(
    Y_in, U_in, V_in,
    meta_in: Dict,
    out_dir: str,
    amount: float = 3.0,
    radius: int = 3,
    threshold: float = 0.001,      # Absolute gate in [0,1] (≈1.5/255)
    halo_limit: float = 2.0,
    *,
    local_threshold: bool = False,
    local_thr_radius: int | None = None,
    grad_preblur_radius: int = 0,
    halo_neg_scale: float = 1.0,
    save_preview_rgb: bool = True,
    save_planar_yuv: bool = True,
    save_debug_maps: bool = True,
    filename_stem: str = "yuv444",
    # —— 8-bit friendly gating (mix absolute/LSB/MAD) ——
    k_sigma: float = 0.0,          # Weight for MAD σ in gate
    threshold_lsb: float = 0.05,   # Base gate in LSB units
    thr_min_lsb: float = 0.0,      # Lower clamp in LSB
    thr_max_lsb: float = 2.0       # Upper clamp in LSB
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict, Dict]:
    """Edge enhancement on Y' only, with gating/tone/halo controls; outputs uint8 YUV444 and metadata."""

    os.makedirs(out_dir, exist_ok=True)

    # Log inputs
    try:
        logger.info("[EE] In dtypes: Y=%s U=%s V=%s | shape=%dx%d",
                    str(getattr(Y_in, "dtype", "?")),
                    str(getattr(U_in, "dtype", "?")),
                    str(getattr(V_in, "dtype", "?")),
                    int(Y_in.shape[0]), int(Y_in.shape[1]))
    except Exception:
        pass

    # Normalize to [0,1]
    in_dtype = getattr(Y_in, "dtype", None)
    Y = _sanitize01(Y_in); U = _sanitize01(U_in); V = _sanitize01(V_in)
    assert Y.ndim == U.ndim == V.ndim == 2 and Y.shape == U.shape == V.shape
    H, W = Y.shape

    # 1) USM high-pass
    base = _blur_gauss_box(Y, radius=max(int(radius), 1), passes=2)
    hp = Y - base

    # 2) Noise gate (abs + k_sigma*σ), clamped to [thr_min_lsb, thr_max_lsb]
    sigma_global = _mad_sigma(hp)

    if in_dtype == np.uint8:
        lsb = 1.0 / 255.0
    elif np.issubdtype(in_dtype, np.integer):
        lsb = 1.0 / float(np.iinfo(in_dtype).max)  # e.g., uint16 → 1/65535
    else:
        # Float inputs: treat as display-quantized; default 8-bit unless overridden in meta.
        bitdepth = int(meta_in.get("ee_bitdepth", 8)) if isinstance(meta_in, dict) else 8
        lsb = 1.0 / (2 ** bitdepth - 1)

    thr_abs = max(float(threshold), float(threshold_lsb) * lsb)
    thr_min = float(thr_min_lsb) * lsb
    thr_max = float(thr_max_lsb) * lsb

    thr_eff_global = np.clip(max(thr_abs, float(k_sigma) * float(sigma_global)), thr_min, thr_max)

    if local_threshold:
        r_thr = max(int(local_thr_radius) if local_thr_radius is not None else max(int(radius) * 2, 2), 1)
        sigma_local = _box_filter_2d(np.abs(hp), radius=r_thr)
        thr_map = np.clip(np.maximum(thr_abs, float(k_sigma) * sigma_local), thr_min, thr_max)
        gate = _smoothstep(thr_map, 3.0 * thr_map, np.abs(hp))
    else:
        gate = _smoothstep(thr_eff_global, 3.0 * thr_eff_global, np.abs(hp))

    hp_g = hp * gate

    # 3) Edge mask
    g2 = _grad_energy(Y, preblur_radius=int(grad_preblur_radius))
    g = np.sqrt(g2 + _EPS)
    g50 = _safe_quantile(g, 0.50); g95 = _safe_quantile(g, 0.95)
    edge = _smoothstep(g50, max(g95, g50 + 1e-6), g)

    # 4) Tone protection (shadows/highlights)
    shL, shH = 0.12, 0.22
    hlL, hlH = 0.82, 0.94
    shadow_w = _smoothstep(shL, shH, Y)
    highlight_w = 1.0 - _smoothstep(hlL, hlH, Y)
    tone_w = shadow_w * highlight_w

    # 5) Initial gain
    gain = float(amount) * edge * tone_w

    # 6) Asymmetric halo limiter
    local_contrast = _box_filter_2d(np.abs(hp), radius=max(int(radius), 1))
    cap_pos = float(halo_limit) * (local_contrast + 1e-4)
    cap_neg = float(halo_limit) * float(halo_neg_scale) * (local_contrast + 1e-4)

    delta = gain * hp_g
    delta = _box_filter_2d(delta, radius=1)
    delta = np.minimum(np.maximum(delta, -cap_neg), cap_pos)

    Y_out = np.clip(Y + delta, 0.0, 1.0)
    U_out = np.clip(U, 0.0, 1.0)
    V_out = np.clip(V, 0.0, 1.0)

    # ---------- Stats ----------
    info = dict(
        amount=float(amount),
        radius=int(radius),
        threshold=float(threshold),
        threshold_lsb=float(threshold_lsb),
        thr_min_lsb=float(thr_min_lsb),
        thr_max_lsb=float(thr_max_lsb),
        thr_abs=float(thr_abs),
        k_sigma=float(k_sigma),
        halo_limit=float(halo_limit),
        halo_neg_scale=float(halo_neg_scale),
        grad_method="central_diff",
        grad_preblur_radius=int(grad_preblur_radius),
        local_threshold=bool(local_threshold),
        local_thr_radius=int(local_thr_radius) if local_thr_radius is not None else max(int(radius) * 2, 2),
        sigma_est_global=float(sigma_global),
        thr_eff_global=float(thr_eff_global),
        g50=float(g50), g95=float(g95),
        image_shape=[int(H), int(W)],
        saved={}
    )
    info["sigma_est"] = info["sigma_est_global"]
    info["thr_eff"] = info["thr_eff_global"]

    if local_threshold and 'thr_map' in locals():
        tv = thr_map[np.isfinite(thr_map)]
        if tv.size > 0:
            info["thr_local_stats"] = dict(
                min=float(np.min(tv)), med=float(np.median(tv)), max=float(np.max(tv))
            )

    # Delta in LSB
    try:
        y_in_u8 = _to_uint8(Y) if in_dtype != np.uint8 else np.asarray(Y_in, np.uint8)
        y_out_u8 = _to_uint8(Y_out)
        diff = np.abs(y_out_u8.astype(np.int16) - y_in_u8.astype(np.int16))
        info["delta_mean_lsb"] = float(diff.mean())
        info["delta_max_lsb"] = int(diff.max())
        info["delta_changed_pct_ge1"] = float((diff >= 1).mean() * 100.0)
    except Exception:
        pass

    # ---------- Outputs ----------
    try:
        y8 = _to_uint8(Y_out); u8 = _to_uint8(U_out); v8 = _to_uint8(V_out)
        Image.fromarray(y8).save(os.path.join(out_dir, f"{filename_stem}_ee_Y.png"))
        Image.fromarray(u8).save(os.path.join(out_dir, f"{filename_stem}_ee_U.png"))
        Image.fromarray(v8).save(os.path.join(out_dir, f"{filename_stem}_ee_V.png"))
        yuv_path = None
        if save_planar_yuv:
            yuv_path = os.path.join(out_dir, f"{filename_stem}_ee.yuv")
            with open(yuv_path, "wb") as f:
                f.write(y8.tobytes(order="C")); f.write(u8.tobytes(order="C")); f.write(v8.tobytes(order="C"))
        rng = (meta_in.get("range") or meta_in.get("tmc_range") or meta_in.get("csc_range") or "full").lower() if isinstance(meta_in, dict) else "full"
        info["saved"]["png"] = {
            "Y": os.path.join(out_dir, f"{filename_stem}_ee_Y.png"),
            "U": os.path.join(out_dir, f"{filename_stem}_ee_U.png"),
            "V": os.path.join(out_dir, f"{filename_stem}_ee_V.png"),
        }
        if yuv_path:
            info["saved"]["raw"] = dict(path=yuv_path, layout="planar", order="Y then U then V",
                                        shape=[H, W], format="YUV444", range=rng)
    except Exception as e:
        info.setdefault("errors", []).append(f"save_yuv_png_raw: {repr(e)}")
        logger.warning("[EE] 保存 YUV/PNG 失败：%s", e)

    if save_debug_maps:
        try:
            _save_gray_png(os.path.join(out_dir, f"{filename_stem}_ee_edge.png"), edge, True)
            _save_gray_png(os.path.join(out_dir, f"{filename_stem}_ee_gate.png"), gate, True)
            _save_gray_png(os.path.join(out_dir, f"{filename_stem}_ee_tone.png"), tone_w, True)
            _save_gray_png(os.path.join(out_dir, f"{filename_stem}_ee_local_contrast.png"), local_contrast, False)
            _save_gray_png(os.path.join(out_dir, f"{filename_stem}_ee_delta_vis.png"), delta, False)
        except Exception as e:
            info.setdefault("errors", []).append(f"save_debug_maps: {repr(e)}")
            logger.warning("[EE] 保存调试可视化失败：%s", e)

    if bool(save_preview_rgb):
        try:
            rgb8 = _yuv2rgb_preview(Y_out, U_out, V_out, meta_in if isinstance(meta_in, dict) else {})
            if rgb8 is not None:
                p = os.path.join(out_dir, f"{filename_stem}_ee_preview_rgb.png")
                Image.fromarray(rgb8).save(p)
                info["saved"]["preview_rgb"] = dict(
                    path=p,
                    range=str((meta_in.get("range") or meta_in.get("tmc_range") or meta_in.get("csc_range") or "full") if isinstance(meta_in, dict) else "full").lower(),
                    uv_center=float(meta_in.get("uv_center", 0.5)) if isinstance(meta_in, dict) else 0.5,
                    has_coeffs=bool(isinstance(meta_in, dict) and isinstance((meta_in.get("coeffs") or meta_in.get("csc_coeffs")), dict))
                )
        except Exception as e:
            info.setdefault("errors", []).append(f"save_preview_rgb: {repr(e)}")
            logger.warning("[EE] 预览保存失败：%s", e)

    # info.json
    try:
        with open(os.path.join(out_dir, "ee_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("[EE] 写入 ee_info.json 失败：%s", e)

    # Meta out (record params without altering input structure)
    meta_out = dict(meta_in) if isinstance(meta_in, dict) else {}
    meta_out["ee_after_tmc"] = True
    meta_out["ee_params"] = dict(
        amount=float(amount), radius=int(radius),
        threshold=float(threshold), threshold_lsb=float(threshold_lsb),
        thr_min_lsb=float(thr_min_lsb), thr_max_lsb=float(thr_max_lsb),
        halo_limit=float(halo_limit), local_threshold=bool(local_threshold),
        local_thr_radius=int(local_thr_radius) if local_thr_radius is not None else max(int(radius) * 2, 2),
        grad_preblur_radius=int(grad_preblur_radius), halo_neg_scale=float(halo_neg_scale),
        k_sigma=float(k_sigma)
    )

    y8 = _to_uint8(Y_out); u8 = _to_uint8(U_out); v8 = _to_uint8(V_out)
    logger.info("[EE] Out dtypes: Y=%s U=%s V=%s", y8.dtype, u8.dtype, v8.dtype)
    return (y8, u8, v8), meta_out, info
