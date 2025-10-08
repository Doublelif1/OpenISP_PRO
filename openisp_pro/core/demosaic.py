# -*- coding: utf-8 -*-
"""
demosaic.py — color-fidelity / brightness-preserving version.
Single public API: run_demosaic_raw().

Highlights:
- Color-difference method (R-G / B-G) with bilinear fill; no extra chroma suppression.
- Optional very light USM on luma only; multiplies RGB uniformly to keep color ratios.
- Brightness back-fill: align demosaiced G mean to the mosaic G mean after AWB.
- All previews map by meta["white_level"] then sRGB OETF (same as pipeline).

Parameters (aligned with your pipeline):
- bayer:         "RGGB"/"GRBG"/"GBRG"/"BGGR"
- mhc_only:      Kept for compatibility; False runs full pipeline (default)
- k_sigma:       Placeholder (unused)
- moire_chroma_atten:  Chroma diff scale, 1.0 = no suppression (most color-preserving)
- gf_radius:     Small box blur radius for chroma diffs (0/1/2); default 1 (very light)
- gf_eps_coeff:  Placeholder (unused)
- ratio_min/max: Clamp R/G and B/G ratios (wide by default to avoid gray drift)
- usm_amount:    Luma-only USM strength (0–0.5 recommended, default 0.15; 0 disables)
- usm_radius:    USM blur radius (1–2)
- usm_thresh:    USM threshold in linear domain, relative to WL (0–1; default 0)
- save_plots:    Save PNG previews to out_dir
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image
import json
import logging

logger = logging.getLogger("demosaic")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------
# Basics
# ---------------------------
@dataclass
class BayerMask:
    R: np.ndarray
    Gr: np.ndarray
    Gb: np.ndarray
    B: np.ndarray


def _bayer_masks(h: int, w: int, bayer: str = "RGGB") -> BayerMask:
    """Return boolean masks for R/Gr/Gb/B according to the Bayer pattern."""
    bayer = (bayer or "RGGB").upper()
    R = np.zeros((h, w), dtype=bool)
    Gr = np.zeros((h, w), dtype=bool)
    Gb = np.zeros((h, w), dtype=bool)
    B = np.zeros((h, w), dtype=bool)

    if bayer == "RGGB":
        R[0::2, 0::2] = True
        Gr[0::2, 1::2] = True
        Gb[1::2, 0::2] = True
        B[1::2, 1::2] = True
    elif bayer == "GRBG":
        Gr[0::2, 0::2] = True
        R[0::2, 1::2] = True
        B[1::2, 0::2] = True
        Gb[1::2, 1::2] = True
    elif bayer == "GBRG":
        Gb[0::2, 0::2] = True
        B[0::2, 1::2] = True
        R[1::2, 0::2] = True
        Gr[1::2, 1::2] = True
    elif bayer == "BGGR":
        B[0::2, 0::2] = True
        Gb[0::2, 1::2] = True
        Gr[1::2, 0::2] = True
        R[1::2, 1::2] = True
    else:
        # Fallback to RGGB
        R[0::2, 0::2] = True
        Gr[0::2, 1::2] = True
        Gb[1::2, 0::2] = True
        B[1::2, 1::2] = True
    return BayerMask(R, Gr, Gb, B)


def _srgb_oetf(x01: np.ndarray) -> np.ndarray:
    """Linear → sRGB OETF (preview only)."""
    x = np.clip(x01, 0.0, 1.0).astype(np.float32)
    a = 0.055
    thr = 0.0031308
    y = np.where(x <= thr, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)
    return np.clip(y, 0.0, 1.0)


def _save_preview_rgb(rgb_lin: np.ndarray, wl: float, out_png: str, max_side: int = 1600) -> None:
    """Normalize by WL → sRGB → save PNG (matches the pipeline viewing)."""
    wl = float(wl) if wl and wl > 0 else 1.0
    rgb01 = np.clip(rgb_lin.astype(np.float32) / wl, 0.0, 1.0)
    srgb01 = _srgb_oetf(rgb01)
    img8 = (srgb01 * 255.0 + 0.5).astype(np.uint8)

    if max_side and max_side > 0:
        h, w = img8.shape[:2]
        m = max(h, w)
        if m > max_side:
            scale = max_side / float(m)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            _resampling = getattr(Image, "Resampling", Image)
            img8 = np.asarray(Image.fromarray(img8).resize((new_w, new_h), resample=_resampling.BICUBIC), dtype=np.uint8)
    Image.fromarray(img8, mode="RGB").save(out_png)


def _shift_copy(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Zero-padded shift (no wraparound)."""
    H, W = a.shape
    out = np.zeros_like(a)
    y0 = max(0, dy)
    y1 = H + min(0, dy)
    x0 = max(0, dx)
    x1 = W + min(0, dx)
    out[y0:y1, x0:x1] = a[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def _bilinear_from_mask(sparse: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Weighted bilinear fill with a 3x3 cross kernel.
    Kernel weights: center×4, 4-neighbors×1 (equiv. [[0,1,0],[1,4,1],[0,1,0]]).
    """
    sparse = sparse.astype(np.float32)
    m = mask.astype(np.float32)

    w = 4.0 * m \
        + _shift_copy(m, 0, 1) + _shift_copy(m, 0, -1) + _shift_copy(m, 1, 0) + _shift_copy(m, -1, 0)
    v = 4.0 * sparse \
        + _shift_copy(sparse, 0, 1) + _shift_copy(sparse, 0, -1) + _shift_copy(sparse, 1, 0) + _shift_copy(sparse, -1, 0)

    eps = 1e-8
    out = np.where(w > eps, v / (w + eps), 0.0).astype(np.float32)
    # Keep known samples unchanged.
    out = np.where(mask, sparse, out)
    return out


def _box_blur(x: np.ndarray, r: int) -> np.ndarray:
    """Small-radius box filter (r=0 returns input)."""
    if r <= 0:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    H, W = x.shape[:2]
    out = x.copy()
    # Simple separable approximation; fine for tiny r.
    for _ in range(2):
        # Horizontal
        tmp = np.cumsum(x, axis=1, dtype=np.float32)
        left = np.concatenate([np.zeros((H, 1), np.float32), tmp[:, :-1]], axis=1)
        right = np.concatenate([tmp[:, r:], np.repeat(tmp[:, -1:], r, axis=1)], axis=1)
        win = right - left
        count = np.minimum(np.arange(1, W + 1), r + 1)
        count = np.minimum(count, np.arange(W, 0, -1))
        count = np.minimum(count, 2 * r + 1).astype(np.float32)
        out = win / count[None, :]

        # Vertical
        tmp = np.cumsum(out, axis=0, dtype=np.float32)
        top = np.concatenate([np.zeros((1, W), np.float32), tmp[:-1, :]], axis=0)
        bottom = np.concatenate([tmp[r:, :], np.repeat(tmp[-1:, :], r, axis=0)], axis=0)
        win = bottom - top
        count = np.minimum(np.arange(1, H + 1), r + 1)
        count = np.minimum(count, np.arange(H, 0, -1))
        count = np.minimum(count, 2 * r + 1).astype(np.float32)
        out = win / count[:, None]
        x = out
    return out.astype(np.float32)


def _safe_clip01(x: np.ndarray) -> np.ndarray:
    """Clip to [0,1] as float32."""
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ---------------------------
# Core: color-difference demosaic (color-preserving)
# ---------------------------
def _demosaic_color_difference(
    raw_awb_lin: np.ndarray,
    bayer: str = "RGGB",
    moire_chroma_atten: float = 1.0,
    gf_radius: int = 1,
    ratio_min: float = 0.05,
    ratio_max: float = 16.0,
) -> np.ndarray:
    """
    Color-difference approach:
      1) Build dense G by inserting known G sites then bilinear fill.
      2) Form sparse (R−G)/(B−G), bilinear to full maps.
      3) Reconstruct R = G + (R−G), B = G + (B−G).
      4) Light box blur (r≤1/2) on chroma diffs only.
      5) Clamp wide R/G and B/G ratios to avoid outliers without de-saturating.
    """
    H, W = raw_awb_lin.shape[:2]
    m = _bayer_masks(H, W, bayer)

    # 1) Dense G (merge Gr/Gb).
    G_sparse = np.zeros((H, W), np.float32)
    G_mask = np.zeros((H, W), np.bool_)
    G_sparse[m.Gr] = raw_awb_lin[m.Gr]
    G_sparse[m.Gb] = raw_awb_lin[m.Gb]
    G_mask[m.Gr] = True
    G_mask[m.Gb] = True
    G = _bilinear_from_mask(G_sparse, G_mask)

    # 2) Sparse R and B channels → chroma diffs.
    R_sparse = np.zeros((H, W), np.float32)
    B_sparse = np.zeros((H, W), np.float32)
    R_mask = m.R
    B_mask = m.B
    R_sparse[R_mask] = raw_awb_lin[R_mask]
    B_sparse[B_mask] = raw_awb_lin[B_mask]

    dR_sparse = np.zeros((H, W), np.float32)
    dB_sparse = np.zeros((H, W), np.float32)
    dR_sparse[R_mask] = R_sparse[R_mask] - G[R_mask]
    dB_sparse[B_mask] = B_sparse[B_mask] - G[B_mask]

    dR = _bilinear_from_mask(dR_sparse, R_mask)
    dB = _bilinear_from_mask(dB_sparse, B_mask)

    # 3) Optional chroma attenuation (1.0 keeps color best).
    dR *= float(moire_chroma_atten)
    dB *= float(moire_chroma_atten)

    # 4) Tiny smoothing on chroma diffs only.
    if gf_radius and gf_radius > 0:
        dR = _box_blur(dR, int(gf_radius))
        dB = _box_blur(dB, int(gf_radius))

    R = (G + dR).astype(np.float32)
    B = (G + dB).astype(np.float32)

    # 5) Wide ratio clamps to prevent extreme values without forcing gray.
    eps = 1e-6
    ratio_R = R / np.maximum(G, eps)
    ratio_B = B / np.maximum(G, eps)
    ratio_R = np.clip(ratio_R, float(ratio_min), float(ratio_max))
    ratio_B = np.clip(ratio_B, float(ratio_min), float(ratio_max))
    R = ratio_R * G
    B = ratio_B * G

    rgb = np.stack([R, G, B], axis=-1)
    return rgb.astype(np.float32)


def _luma_only_usm_inplace(rgb: np.ndarray, wl: float, amount: float = 0.15, radius: int = 1, thresh: float = 0.0):
    """
    Very light USM on luma only; scale RGB equally per-pixel (keep color ratios).
    - amount: 0..0.5 recommended; 0 disables
    - radius: 1..2
    - thresh: linear threshold relative to WL (0..1)
    """
    if amount is None or amount <= 0:
        return

    wl = float(wl) if wl and wl > 0 else 1.0
    Y = rgb[..., 1].astype(np.float32)  # Use G as linear luma; stable and neutral.
    Y_blur = _box_blur(Y, max(int(radius), 1))
    detail = Y - Y_blur

    thr = float(thresh) * wl
    if thr > 0:
        mask = (np.abs(detail) > thr).astype(np.float32)
        detail = detail * mask

    Yp = Y + float(amount) * detail
    # Compute per-pixel gain to avoid altering color ratios.
    eps = 1e-6
    gain = np.clip(Yp / np.maximum(Y, eps), 0.5, 2.0).astype(np.float32)
    rgb *= gain[..., None]


def _brightness_backfill(rgb: np.ndarray, raw_cfa: np.ndarray, bayer: str) -> float:
    """
    Brightness back-fill: scale so demosaiced G mean ≈ mosaic G mean.
    Returns the applied scale factor s.
    """
    m = _bayer_masks(*raw_cfa.shape, bayer)
    Gr_mean = float(raw_cfa[m.Gr].mean()) if np.any(m.Gr) else 0.0
    Gb_mean = float(raw_cfa[m.Gb].mean()) if np.any(m.Gb) else 0.0
    Gmosaic = 0.5 * (Gr_mean + Gb_mean)

    Gfull = float(rgb[..., 1].mean())
    eps = 1e-6
    s = (Gmosaic + eps) / (Gfull + eps) if Gfull > 0 else 1.0
    rgb *= s
    return float(s)


# ---------------------------
# Public API
# ---------------------------
def run_demosaic_raw(
    raw_awb_lin: np.ndarray,
    meta_in: Dict[str, Any],
    out_dir: str,
    *,
    bayer: str = "RGGB",
    mhc_only: bool = False,        # Compatibility placeholder; False = full pipeline
    k_sigma: float = 4.0,          # Placeholder
    moire_chroma_atten: float = 1.0,
    gf_radius: int = 1,
    gf_eps_coeff: float = 1.0,     # Placeholder
    ratio_min: float = 0.05,
    ratio_max: float = 16.0,
    usm_amount: float = 0.15,
    usm_radius: int = 1,
    usm_thresh: float = 0.0,
    save_plots: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    RAW → RGB demosaic prioritizing color fidelity.
    Returns: rgb_lin (float32, linear in WL), meta_out, info(dict)
    """
    assert raw_awb_lin.ndim == 2, "run_demosaic_raw expects 2D Bayer RAW (after AWB)."
    H, W = raw_awb_lin.shape[:2]
    bayer = (bayer or meta_in.get("bayer", "RGGB") or "RGGB").upper()

    # Metadata
    meta_out = dict(meta_in or {})
    wl = float(meta_out.get("white_level", 4095.0))
    meta_out["white_level"] = wl
    meta_out["shape_in"] = (H, W)
    meta_out["demosaic"] = "color_difference_preserve_chroma"
    meta_out["bayer"] = bayer

    os.makedirs(out_dir, exist_ok=True)

    # Log mosaic means (for brightness back-fill and sanity).
    m = _bayer_masks(H, W, bayer)
    Rm = float(raw_awb_lin[m.R].mean()) if np.any(m.R) else 0.0
    Grm = float(raw_awb_lin[m.Gr].mean()) if np.any(m.Gr) else 0.0
    Gbm = float(raw_awb_lin[m.Gb].mean()) if np.any(m.Gb) else 0.0
    Bm = float(raw_awb_lin[m.B].mean()) if np.any(m.B) else 0.0
    Gm = 0.5 * (Grm + Gbm)
    logger.info("[MHC/means] R=%.3f  G=%.3f  B=%.3f  |  R/G=%.2f  B/G=%.2f",
                Rm, Gm, Bm, (Rm / max(Gm, 1e-6)), (Bm / max(Gm, 1e-6)))

    # 1) Color-difference demosaic (color-preserving).
    rgb = _demosaic_color_difference(
        raw_awb_lin,
        bayer=bayer,
        moire_chroma_atten=float(moire_chroma_atten),
        gf_radius=int(gf_radius),
        ratio_min=float(ratio_min),
        ratio_max=float(ratio_max),
    )

    # 2) Brightness back-fill to avoid darkening.
    s_bright = _brightness_backfill(rgb, raw_awb_lin, bayer)
    logger.info("[DEMOSAIC/bright-backfill] scale=%.6f (align G_mean to mosaic G_mean)", s_bright)

    # 3) Luma-only USM (keeps color ratios; set usm_amount=0 to disable).
    _luma_only_usm_inplace(rgb, wl=wl, amount=float(usm_amount), radius=int(usm_radius), thresh=float(usm_thresh))

    # 4) Clip to [0, WL] (linear).
    rgb = np.clip(rgb, 0.0, wl).astype(np.float32)

    # Info / metrics
    info: Dict[str, Any] = {}
    info["algo"] = "color_difference_preserve_chroma"
    info["params"] = dict(
        bayer=bayer,
        moire_chroma_atten=float(moire_chroma_atten),
        gf_radius=int(gf_radius),
        ratio_min=float(ratio_min),
        ratio_max=float(ratio_max),
        usm_amount=float(usm_amount),
        usm_radius=int(usm_radius),
        usm_thresh=float(usm_thresh),
        bright_backfill_scale=float(s_bright),
    )

    # Quick chroma energy to check “color loss”.
    def _chroma_energy(xrgb: np.ndarray) -> float:
        R, G, B = xrgb[..., 0], xrgb[..., 1], xrgb[..., 2]
        Cr = R - G
        Cb = B - G
        return float(np.mean(Cr * Cr + Cb * Cb))

    info["chroma_energy"] = _chroma_energy(rgb)

    # Previews / artifacts
    if save_plots:
        try:
            _save_preview_rgb(rgb, wl, os.path.join(out_dir, "preview_srgb.png"))
        except Exception as e:
            logger.warning("[DEMOSAIC] preview save failed: %s", e)

        # Optional: export mosaic seeds for visual comparison.
        try:
            seed_dbg = np.stack([raw_awb_lin * m.R, raw_awb_lin * (m.Gr | m.Gb), raw_awb_lin * m.B], axis=-1)
            _save_preview_rgb(seed_dbg, wl, os.path.join(out_dir, "preview_seed_mosaic_rgb.png"))
        except Exception:
            pass

        # Save JSON params.
        try:
            with open(os.path.join(out_dir, "demosaic_info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Meta
    meta_out["shape_out"] = (H, W, 3)
    meta_out["channels"] = 3

    return rgb, meta_out, info
