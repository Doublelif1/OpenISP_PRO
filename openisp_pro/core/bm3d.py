# -*- coding: utf-8 -*-
"""
BM3D (pure NumPy) — Denoising in RAW/Bayer domain (BM3D only).
- Input: linear RAW from BLC (values are counts above BL).
- Process each Bayer sub-channel independently (R/Gr/Gb/B).
- Auto noise sigma via robust MAD; or accept scalar/per-channel input.
- Pure NumPy BM3D (Basic+Wiener) with vectorized fast kernels.
- Output: denoised linear RAW (same size), updated meta, and info.
"""

import os
import json
import numpy as np
import logging
from math import cos, pi
from numpy.lib.stride_tricks import as_strided  # view-based sliding windows to avoid copies

logger = logging.getLogger(__name__)

# ---------- Bayer layout ----------
# Map Bayer mosaic (RGGB/GRBG/GBRG/BGGR) to (row_offset, col_offset).
_BAYER_OFFSETS = {
    "RGGB": ((0,0), (0,1), (1,0), (1,1)),  # R, Gr, Gb, B
    "GRBG": ((0,1), (0,0), (1,1), (1,0)),
    "GBRG": ((1,0), (1,1), (0,0), (0,1)),
    "BGGR": ((1,1), (1,0), (0,1), (0,0)),
}

def _norm_bayer(pat: str) -> str:
    # Normalize/validate Bayer string.
    pat = (pat or "RGGB").upper()
    if pat not in _BAYER_OFFSETS:
        raise ValueError(f"Unsupported bayer pattern: {pat}")
    return pat


# ---------- Split/merge, preview, saving ----------
def _split_bayer_channels(raw_lin: np.ndarray, bayer: str):
    # Split single-channel RAW into four Bayer planes: R/Gr/Gb/B.
    pat = _norm_bayer(bayer)
    (yR,xR),(yGr,xGr),(yGb,xGb),(yB,xB) = _BAYER_OFFSETS[pat]
    return {
        "R":  raw_lin[yR::2,  xR::2],
        "Gr": raw_lin[yGr::2, xGr::2],
        "Gb": raw_lin[yGb::2, xGb::2],
        "B":  raw_lin[yB::2,  xB::2],
    }

def _merge_bayer_channels(chans: dict, shape_hw: tuple, bayer: str):
    # Merge four planes back to single-channel RAW.
    h, w = shape_hw
    pat = _norm_bayer(bayer)
    (yR,xR),(yGr,xGr),(yGb,xGb),(yB,xB) = _BAYER_OFFSETS[pat]
    out = np.empty((h, w), dtype=chans["R"].dtype)
    out[yR::2,  xR::2]  = chans["R"]
    out[yGr::2, xGr::2] = chans["Gr"]
    out[yGb::2, xGb::2] = chans["Gb"]
    out[yB::2,  xB::2]  = chans["B"]
    return out

def _to_u8(img01, black_q=0.01, white_q=0.999):
    # Quantile-based 0..1 stretch to 8-bit for quick preview.
    x = np.asarray(img01, np.float32)
    lo = float(np.quantile(x, black_q)) if x.size else 0.0
    hi = float(np.quantile(x, white_q)) if x.size else 1.0
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x*255.0 + 0.5).astype(np.uint8)

def _save_exact16(path, img, wl_channel):
    """Save linear RAW into 16-bit single-channel PNG (linear; for regression/compare)."""
    try:
        import imageio.v2 as iio
        wl_eff = int(wl_channel) if wl_channel is not None else int(np.max(img) if img.size else 1)
        scale = 65535.0 / max(wl_eff, 1)
        arr16 = np.clip(img, 0, wl_eff) * scale
        iio.imwrite(str(path), np.rint(arr16).astype(np.uint16))
    except Exception as e:
        logger.debug("[BM3D] exact16 save skipped: %s", e)

def _save_u8(path, img01):
    # Save 8-bit preview; prefer cv2, fallback to imageio/matplotlib.
    try:
        import cv2
        cv2.imwrite(str(path), _to_u8(img01))
    except Exception:
        try:
            import imageio.v2 as iio
            iio.imwrite(str(path), _to_u8(img01))
        except Exception:
            import matplotlib.pyplot as plt
            plt.imsave(str(path), _to_u8(img01), cmap='gray', vmin=0, vmax=1)


# ---------- Sigma estimation / normalization ----------
def _robust_sigma_mad(x: np.ndarray) -> float:
    """
    Robust noise estimation via neighbor differences + MAD.
    Convert MAD to sigma and divide by sqrt(2) to counter diff variance ≈ 2σ².
    """
    x = np.asarray(x, np.float32)
    if x.size == 0:
        return 0.0
    dh = x[:, 1:] - x[:, :-1]
    dv = x[1:, :] - x[:-1, :]
    def _mad_to_sigma(d):
        mad = np.median(np.abs(d - np.median(d)))
        return (mad / 0.67448975) / np.sqrt(2.0 + 1e-12)
    sh = _mad_to_sigma(dh) if dh.size else 0.0
    sv = _mad_to_sigma(dv) if dv.size else 0.0
    s = 0.5*(sh + sv)
    if not np.isfinite(s):
        s = 0.0
    return float(max(s, 0.0))

def _to01_per_channel(ch: np.ndarray, denom: float) -> np.ndarray:
    # Map per-plane linear values to 0..1 by its remaining dynamic range.
    denom = max(float(denom), 1.0)
    return np.clip(np.asarray(ch, np.float32) / denom, 0.0, 1.0)

def _from01_per_channel(ch01: np.ndarray, denom: float) -> np.ndarray:
    # Restore from 0..1 to linear counts.
    denom = max(float(denom), 1.0)
    return np.asarray(ch01, np.float32) * denom


# =========================  Pure NumPy BM3D params & bases  =========================
class _BM3DParams:
    """
    Core BM3D hyperparams: patch size/step, search window, group size,
    thresholds, match scale, and Kaiser window shape.
    """
    def __init__(self,
                 patch_size=8,
                 step=4,
                 search_window=25,
                 max_group_size=16,
                 hard_thr_lambda=2.7,
                 match_tau_mult=1800.0,
                 kaiser_beta=2.0):
        self.patch_size = int(patch_size)
        self.step = int(step)
        self.search_window = int(search_window)
        self.max_group_size = int(max_group_size)
        self.hard_thr_lambda = float(hard_thr_lambda)
        self.match_tau_mult = float(match_tau_mult)
        self.kaiser_beta = float(kaiser_beta)

def _dct_matrix(n: int) -> np.ndarray:
    """Orthogonal DCT-II matrix (for 2D patch transforms)."""
    C = np.zeros((n, n), dtype=np.float32)
    alpha0 = np.sqrt(1.0/n)
    alpha  = np.sqrt(2.0/n)
    for k in range(n):
        for m in range(n):
            C[k, m] = (alpha0 if k == 0 else alpha) * cos(pi*(2*m+1)*k/(2*n))
    return C

def _haar_matrix(n: int) -> np.ndarray:
    """Orthogonal Haar matrix (length must be power of two)."""
    if n & (n-1) != 0:
        raise ValueError("Haar length must be power of two.")
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        h = H / np.sqrt(2.0)
        H = np.block([[h, h],[h, -h]])
    return H

def _kaiser2d(n, beta) -> np.ndarray:
    """2D Kaiser window for smooth patch aggregation."""
    w1d = np.kaiser(n, beta).astype(np.float32)
    return np.outer(w1d, w1d).astype(np.float32)

def _get_patch(img, y, x, ps):
    # Extract a ps×ps patch.
    return img[y:y+ps, x:x+ps]

def _ssd(a, b):
    # Sum of squared differences.
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sum(d*d))

def _clip_search_bounds(h, w, y, x, ps, sw):
    # Clip search window to image bounds.
    r = sw // 2
    y0 = max(0, y - r); y1 = min(h - ps, y + r)
    x0 = max(0, x - r); x1 = min(w - ps, x + r)
    return y0, y1, x0, x1

def _dct2(patch, C):
    # 2D DCT: Y = C @ X @ C.T
    return C @ patch @ C.T

def _idct2(coeff, C):
    # Inverse 2D DCT.
    return C.T @ coeff @ C

def _haar1d_along_groups(group, H):
    # 1D Haar along group dimension (k) after stacking similar patches.
    ps, _, k = group.shape
    g = group.reshape(ps*ps, k).astype(np.float32)
    return (g @ H.T).reshape(ps, ps, H.shape[0])

def _ihaar1d_along_groups(group_t, H):
    # Inverse Haar along group dimension.
    ps2, k = group_t.reshape(-1, group_t.shape[2]).shape
    ps = int(np.sqrt(ps2))
    g = group_t.reshape(ps*ps, group_t.shape[2])
    return (g @ H).reshape(ps, ps, H.shape[0])


# =========================  Readable (naive) BM3D  =========================
def _bm3d_gray_numpy(image: np.ndarray, sigma: float, p: _BM3DParams = _BM3DParams()):
    """
    Naive BM3D in two stages:
      Basic (hard-thresholding) then Wiener (pilot-guided filtering).
    Returns u_final (denoised) and u_basic (pilot).
    """
    assert image.ndim == 2, "bm3d expects 2D grayscale"
    img = image.astype(np.float32).copy()
    h, w = img.shape
    ps = p.patch_size
    sw = p.search_window
    K  = p.max_group_size
    step = p.step

    C = _dct_matrix(ps)
    kaiser = _kaiser2d(ps, p.kaiser_beta)

    def _haar_for_k(k):
        # Round group length up to power of two for Haar.
        n = 1
        while n < k: n <<= 1
        return _haar_matrix(n), n

    # --- Stage 1: Basic (hard threshold) ---
    basic_num = np.zeros_like(img, dtype=np.float32)
    basic_den = np.zeros_like(img, dtype=np.float32)
    tau_match = p.match_tau_mult * (sigma*sigma) / (ps*ps)

    for y in range(0, h-ps+1, step):
        for x in range(0, w-ps+1, step):
            ref = _get_patch(img, y, x, ps)

            # Search similar patches in window
            y0,y1,x0,x1 = _clip_search_bounds(h, w, y, x, ps, sw)
            cand = []
            for yy in range(y0, y1+1):
                for xx in range(x0, x1+1):
                    if yy == y and xx == x:
                        cand.append((0.0, yy, xx))
                        continue
                    pch = _get_patch(img, yy, xx, ps)
                    d = _ssd(ref, pch)/(ps*ps)
                    if d <= tau_match:
                        cand.append((d, yy, xx))

            cand.sort(key=lambda t: t[0])
            if len(cand) == 0:
                cand = [(0.0, y, x)]
            if len(cand) > K:
                cand = cand[:K]
            k = len(cand)

            group = np.zeros((ps, ps, k), dtype=np.float32)
            for i, (_, yy, xx) in enumerate(cand):
                group[:,:,i] = _get_patch(img, yy, xx, ps)

            # 2D DCT + Haar (pad to power of two)
            for i in range(k):
                group[:,:,i] = _dct2(group[:,:,i], C)
            Hk, nk = _haar_for_k(k)
            if nk != k:
                tmp = np.zeros((ps, ps, nk), dtype=np.float32)
                tmp[:,:,:k] = group
                group = tmp
            group_t = _haar1d_along_groups(group, Hk)

            # Hard threshold
            N = ps*ps*group_t.shape[2]
            T = p.hard_thr_lambda * sigma * np.sqrt(2.0*np.log(max(N, 2)))
            mask = np.abs(group_t) > T
            group_t_thr = group_t * mask

            # Inverse transforms
            group_rec = _ihaar1d_along_groups(group_t_thr, Hk)[:,:,:k]
            for i in range(k):
                group_rec[:,:,i] = _idct2(group_rec[:,:,i], C)

            # Aggregate with Kaiser; weight ~ 1/nnz
            nnz = np.maximum(1, np.sum(mask))
            weight = (kaiser / float(nnz)).astype(np.float32)
            for i, (_, yy, xx) in enumerate(cand):
                basic_num[yy:yy+ps, xx:xx+ps] += group_rec[:,:,i] * weight
                basic_den[yy:yy+ps, xx:xx+ps] += weight

    u_basic = np.divide(basic_num, np.maximum(1e-8, basic_den), dtype=np.float32)

    # --- Stage 2: Wiener (pilot-guided) ---
    final_num = np.zeros_like(img, dtype=np.float32)
    final_den = np.zeros_like(img, dtype=np.float32)

    for y in range(0, h-ps+1, step):
        for x in range(0, w-ps+1, step):
            ref_p = _get_patch(u_basic, y, x, ps)
            y0,y1,x0,x1 = _clip_search_bounds(h, w, y, x, ps, sw)

            cand = []
            for yy in range(y0, y1+1):
                for xx in range(x0, x1+1):
                    pch = _get_patch(u_basic, yy, xx, ps)
                    d = _ssd(ref_p, pch)/(ps*ps)
                    cand.append((d, yy, xx))

            cand.sort(key=lambda t: t[0])
            if len(cand) > K:
                cand = cand[:K]
            k = len(cand)

            noisy_grp = np.zeros((ps, ps, k), dtype=np.float32)
            pilot_grp = np.zeros((ps, ps, k), dtype=np.float32)
            for i, (_, yy, xx) in enumerate(cand):
                noisy_grp[:,:,i] = _get_patch(img, yy, xx, ps)
                pilot_grp[:,:,i] = _get_patch(u_basic, yy, xx, ps)

            for i in range(k):
                noisy_grp[:,:,i] = _dct2(noisy_grp[:,:,i], C)
                pilot_grp[:,:,i] = _dct2(pilot_grp[:,:,i], C)

            Hk, nk = _haar_for_k(k)
            if nk != k:
                tmpn = np.zeros((ps, ps, nk), dtype=np.float32); tmpn[:,:,:k] = noisy_grp
                tmpp = np.zeros((ps, ps, nk), dtype=np.float32); tmpp[:,:,:k] = pilot_grp
                noisy_grp, pilot_grp = tmpn, tmpp

            noisy_t = _haar1d_along_groups(noisy_grp, Hk)
            pilot_t = _haar1d_along_groups(pilot_grp, Hk)

            # Wiener gain G = S^2 / (S^2 + σ^2), where S^2 from pilot.
            S2 = pilot_t**2
            G  = S2 / (S2 + sigma*sigma + 1e-8)
            den_t = noisy_t * G

            den_grp = _ihaar1d_along_groups(den_t, Hk)[:,:,:k]
            for i in range(k):
                den_grp[:,:,i] = _idct2(den_grp[:,:,i], C)

            wiener_weight = (np.mean(G) + 1e-8) * kaiser
            for i, (_, yy, xx) in enumerate(cand):
                final_num[yy:yy+ps, xx:xx+ps] += den_grp[:,:,i] * wiener_weight
                final_den[yy:yy+ps, xx:xx+ps] += wiener_weight

    u_final = np.divide(final_num, np.maximum(final_den, 1e-8), dtype=np.float32)
    return u_final, u_basic


# =========================  Vectorized fast BM3D  =========================
# Same logic as the naive version but batched and window-viewed for speed.
_BM3D_CACHE = {"C": {}, "H": {}, "kaiser": {}}

def _get_C(ps):
    # Cache 2D-DCT matrix by patch size.
    C = _BM3D_CACHE["C"].get(ps)
    if C is None:
        C = _dct_matrix(ps).astype(np.float32)
        _BM3D_CACHE["C"][ps] = C
    return C

def _get_H(nk):
    # Cache Haar matrix by group length (power of two).
    H = _BM3D_CACHE["H"].get(nk)
    if H is None:
        H = _haar_matrix(nk).astype(np.float32)
        _BM3D_CACHE["H"][nk] = H
    return H

def _get_kaiser(ps, beta):
    # Cache Kaiser window by (ps, beta).
    key = (ps, float(beta))
    K = _BM3D_CACHE["kaiser"].get(key)
    if K is None:
        K = _kaiser2d(ps, beta).astype(np.float32)
        _BM3D_CACHE["kaiser"][key] = K
    return K

def _view_patches(img, ps):
    """Zero-copy sliding-window view: (H-ps+1, W-ps+1, ps, ps)."""
    h, w = img.shape
    oh, ow = h - ps + 1, w - ps + 1
    if oh <= 0 or ow <= 0:
        return None
    s0, s1 = img.strides
    return as_strided(img, shape=(oh, ow, ps, ps), strides=(s0, s1, s0, s1))

def _pow2_ceil(n):
    # Round up to next power of two.
    k = 1
    while k < n: k <<= 1
    return k

def _bm3d_gray_numpy_fast(image: np.ndarray, sigma: float, p: _BM3DParams = _BM3DParams()):
    """
    Vectorized BM3D (Basic+Wiener): batch transforms, windowed matching/grouping,
    significant speedup vs. naive implementation.
    """
    assert image.ndim == 2, "bm3d expects 2D grayscale"
    img = image.astype(np.float32, copy=False)
    h, w = img.shape
    ps, sw, K, step = p.patch_size, p.search_window, p.max_group_size, p.step
    if h < ps or w < ps:
        return img.copy(), img.copy()

    C = _get_C(ps)
    kaiser = _get_kaiser(ps, p.kaiser_beta)
    patches = _view_patches(img, ps)     # (Oh, Ow, ps, ps)
    Oh, Ow = patches.shape[:2]

    # Reference positions (controlled by step)
    ref_y = np.arange(0, Oh, step, dtype=np.int32)
    ref_x = np.arange(0, Ow, step, dtype=np.int32)
    r = sw // 2

    # Aggregation buffers
    basic_num = np.zeros_like(img, dtype=np.float32)
    basic_den = np.zeros_like(img, dtype=np.float32)
    final_num = np.zeros_like(img, dtype=np.float32)
    final_den = np.zeros_like(img, dtype=np.float32)

    tau_match = p.match_tau_mult * (sigma * sigma) / float(ps * ps)

    # Batched 2D DCT/IDCT for (ps,ps,k)
    def dct2_batch(G):
        G = np.tensordot(C, G, axes=(1, 0))
        G = np.tensordot(G, C.T, axes=(1, 1)).transpose(0, 2, 1)
        return G

    def idct2_batch(G):
        G = np.tensordot(C.T, G, axes=(1, 0))
        G = np.tensordot(G, C, axes=(1, 1)).transpose(0, 2, 1)
        return G

    # ---------- Basic ----------
    for y0 in ref_y:
        sy0 = max(0, y0 - r); sy1 = min(Oh - 1, y0 + r)
        for x0 in ref_x:
            sx0 = max(0, x0 - r); sx1 = min(Ow - 1, x0 + r)
            ref = patches[y0, x0]
            cand = patches[sy0:sy1+1, sx0:sx1+1]
            Hy, Wx = cand.shape[:2]

            # Avg SSD to select group members
            diff = cand - ref
            dists = np.sum(diff*diff, axis=(2,3), dtype=np.float32) / float(ps*ps)
            dflat = dists.reshape(-1)

            self_idx = (y0 - sy0) * Wx + (x0 - sx0)
            mask = dflat <= tau_match
            mask[self_idx] = True
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                idxs = np.array([self_idx], dtype=np.int64)
            if idxs.size > K:
                part = np.argpartition(dflat[idxs], K-1)[:K]
                idxs = idxs[part]
            idxs = idxs[np.argsort(dflat[idxs])]
            k = int(idxs.size)

            cand_y = idxs // Wx
            cand_x = idxs %  Wx
            group = cand[cand_y, cand_x]                  # (k,ps,ps)
            group = np.transpose(group, (1,2,0)).copy()   # -> (ps,ps,k)

            # 2D DCT + Haar (pad to power of two)
            group = dct2_batch(group)
            nk = _pow2_ceil(k)
            Hk = _get_H(nk)
            if nk != k:
                tmp = np.zeros((ps, ps, nk), dtype=np.float32)
                tmp[:, :, :k] = group
                group = tmp

            g = group.reshape(ps*ps, nk)
            g_t = g @ Hk.T
            group_t = g_t.reshape(ps, ps, nk)

            # Hard threshold
            N = ps*ps*nk
            T = p.hard_thr_lambda * sigma * np.sqrt(2.0*np.log(max(N, 2)))
            mask_nz = (np.abs(group_t) > T)
            group_t_thr = group_t * mask_nz

            # Inverse transforms
            g_rec = (group_t_thr.reshape(ps*ps, nk) @ Hk).reshape(ps, ps, nk)
            g_rec = g_rec[:, :, :k]
            g_rec = idct2_batch(g_rec)

            # Aggregate with Kaiser; weight ~ 1/nnz
            nnz = int(np.sum(mask_nz[:, :, :k])); nnz = max(nnz, 1)
            weight = (kaiser / float(nnz)).astype(np.float32)

            for i in range(k):
                yy = sy0 + int(cand_y[i])
                xx = sx0 + int(cand_x[i])
                basic_num[yy:yy+ps, xx:xx+ps] += g_rec[:, :, i] * weight
                basic_den[yy:yy+ps, xx:xx+ps] += weight

    u_basic = basic_num / np.maximum(basic_den, 1e-8)

    # ---------- Wiener ----------
    patches_noisy = patches
    patches_pilot = _view_patches(u_basic, ps)  # regroup on cleaner pilot
    for y0 in ref_y:
        sy0 = max(0, y0 - r); sy1 = min(Oh - 1, y0 + r)
        for x0 in ref_x:
            sx0 = max(0, x0 - r); sx1 = min(Ow - 1, x0 + r)

            refp = patches_pilot[y0, x0]
            candp = patches_pilot[sy0:sy1+1, sx0:sx1+1]
            Hy, Wx = candp.shape[:2]

            diff = candp - refp
            dists = np.sum(diff*diff, axis=(2,3), dtype=np.float32) / float(ps*ps)
            dflat = dists.reshape(-1)
            if dflat.size > K:
                idxs = np.argpartition(dflat, K-1)[:K]
                idxs = idxs[np.argsort(dflat[idxs])]
            else:
                idxs = np.argsort(dflat)
            k = int(idxs.size)

            cand_y = idxs // Wx
            cand_x = idxs %  Wx

            noisy_grp = patches_noisy[sy0:sy1+1, sx0:sx1+1][cand_y, cand_x]
            pilot_grp = candp[cand_y, cand_x]
            noisy_grp = np.transpose(noisy_grp, (1,2,0)).copy()
            pilot_grp = np.transpose(pilot_grp, (1,2,0)).copy()

            noisy_grp = dct2_batch(noisy_grp)
            pilot_grp = dct2_batch(pilot_grp)

            nk = _pow2_ceil(k)
            Hk = _get_H(nk)
            if nk != k:
                tmpn = np.zeros((ps, ps, nk), dtype=np.float32); tmpn[:, :, :k] = noisy_grp
                tmpp = np.zeros((ps, ps, nk), dtype=np.float32); tmpp[:, :, :k] = pilot_grp
                noisy_grp, pilot_grp = tmpn, tmpp

            n_t = (noisy_grp.reshape(ps*ps, nk) @ Hk.T).reshape(ps, ps, nk)
            p_t = (pilot_grp.reshape(ps*ps, nk) @ Hk.T).reshape(ps, ps, nk)

            # Wiener gain (trust pilot energy).
            S2 = p_t * p_t
            G  = S2 / (S2 + sigma*sigma + 1e-8)
            den_t = n_t * G

            den_grp = (den_t.reshape(ps*ps, nk) @ Hk).reshape(ps, ps, nk)[:, :, :k]
            den_grp = idct2_batch(den_grp)

            g2_mean = float(np.mean(G*G))
            wiener_weight = (1.0 / (1e-8 + (sigma*sigma) * g2_mean)) * kaiser

            for i in range(k):
                yy = sy0 + int(cand_y[i])
                xx = sx0 + int(cand_x[i])
                final_num[yy:yy+ps, xx:xx+ps] += den_grp[:, :, i] * wiener_weight
                final_den[yy:yy+ps, xx:xx+ps] += wiener_weight

    u_final = final_num / np.maximum(final_den, 1e-8)
    return u_final.astype(np.float32), u_basic.astype(np.float32)


# =========================  RAW entry (pipeline hook)  =========================
def run_bm3d_raw(raw_lin: np.ndarray,
                 meta: dict,
                 out_dir: str,
                 *,
                 sigma=None,            # None/scalar/dict (counts above BL)
                 method: str = "auto",  # "auto" | "bm3d" (both map to BM3D here)
                 strength: float = 1.0, # global scale on sigma
                 save_plots: bool = True,
                 save_exact16: bool = False,
                 stage: str = "all",   # kept for backward compat; unused internally
                 sigma_cap01: float = 0.25,      # sigma cap in 0..1 domain
                 sigma_floor_counts: float = 0.0 # sigma floor in counts
                 ):
    """
    Denoise BLC-linear RAW per Bayer sub-channel.
    Returns (raw_dn_lin, meta_out, info).
    """
    # Create output dir only if needed.
    if save_plots or save_exact16:
        os.makedirs(out_dir, exist_ok=True)

    meta = dict(meta)
    bayer = _norm_bayer(meta.get("bayer", "RGGB"))
    WL = float(meta.get("white_level", 4095))
    bl_used = meta.get("black_level_used", {"R":0,"Gr":0,"Gb":0,"B":0})

    # Per-channel remaining dynamic range = WL - BL_used.
    denom = {
        "R":  max(WL - float(bl_used.get("R",  0.0)), 1.0),
        "Gr": max(WL - float(bl_used.get("Gr", 0.0)), 1.0),
        "Gb": max(WL - float(bl_used.get("Gb", 0.0)), 1.0),
        "B":  max(WL - float(bl_used.get("B",  0.0)), 1.0),
    }

    # Split channels.
    raw_lin = np.asarray(raw_lin, np.float32)
    chans = _split_bayer_channels(raw_lin, bayer)

    # Parse/estimate sigma (counts above BL).
    if sigma is None:
        sigma_map = {k: _robust_sigma_mad(ch) for k, ch in chans.items()}
    elif np.isscalar(sigma):
        s = float(sigma); sigma_map = {"R":s, "Gr":s, "Gb":s, "B":s}
    elif isinstance(sigma, dict):
        sigma_map = {
            "R": float(sigma.get("R", 0.0)),
            "Gr": float(sigma.get("Gr", 0.0)),
            "Gb": float(sigma.get("Gb", 0.0)),
            "B": float(sigma.get("B", 0.0)),
        }
    else:
        raise ValueError("sigma must be None/scalar/dict")

    # Backend choice (BM3D only here).
    if method not in ("auto", "bm3d"):
        raise ValueError("method must be 'auto' or 'bm3d'")
    used_backend = "bm3d_numpy_fast"

    # Per-channel BM3D: normalize to 0..1, scale sigma accordingly, run, then restore.
    out_ch = {}
    sigma_used = {}
    for c, ch in chans.items():
        s_lin = max(float(sigma_map[c]) * float(strength), float(sigma_floor_counts))
        sigma_used[c] = s_lin

        ch01 = _to01_per_channel(ch, denom[c])
        s01  = float(s_lin) / float(denom[c])
        s01  = max(min(s01, float(sigma_cap01)), 1e-3)

        den01, _basic = _bm3d_gray_numpy_fast(ch01, sigma=s01, p=_BM3DParams(
            patch_size=8,
            step=8,              # faster than 4 (slight quality drop)
            search_window=17,    # smaller window (faster)
            max_group_size=12,   # smaller group (faster)
            hard_thr_lambda=2.9, # slightly stronger Basic
            match_tau_mult=900.0,# stricter matching
            kaiser_beta=2.0
        ))
        out_ch[c] = _from01_per_channel(den01, denom[c]).astype(np.float32)

    # Merge back to RAW.
    raw_dn_lin = _merge_bayer_channels(out_ch, raw_lin.shape, bayer).astype(np.float32)

    # Save 8-bit previews (shared quantile mapping from input).
    stem = os.path.splitext(os.path.basename(meta.get("file_basename", "image")))[0]
    if save_plots:
        def _whole01(_raw):
            # Normalize each plane by its range and place back for grayscale preview.
            chans2 = _split_bayer_channels(np.asarray(_raw, np.float32), bayer)
            y01 = np.empty_like(_raw, dtype=np.float32)
            for cc, (yoff, xoff) in zip(["R","Gr","Gb","B"], _BAYER_OFFSETS[bayer]):
                y01[yoff::2, xoff::2] = np.clip(chans2[cc] / denom[cc], 0.0, 1.0)
            return y01
        inp01 = _whole01(raw_lin)
        out01 = _whole01(raw_dn_lin)
        lo = float(np.quantile(inp01, 0.01)) if inp01.size else 0.0
        hi = float(np.quantile(inp01, 0.999)) if inp01.size else 1.0
        def _apply(x):
            return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

        in_png  = os.path.join(out_dir, f"{stem}_BM3D_input_8bit.png")
        out_png = os.path.join(out_dir, f"{stem}_BM3D_output_8bit.png")
        _save_u8(in_png,  _apply(inp01))
        _save_u8(out_png, _apply(out01))
        try:
            with open(os.path.join(out_dir, f"{stem}_BM3D_preview_range.json"), "w", encoding="utf-8") as f:
                json.dump({"lo": lo, "hi": hi}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Optional: linear 16-bit PNG for strict regression.
    if save_exact16:
        exact16_in  = os.path.join(out_dir, f"{stem}_BM3D_input_exact16.png")
        exact16_out = os.path.join(out_dir, f"{stem}_BM3D_output_exact16.png")
        _save_exact16(exact16_in,  raw_lin,    WL)
        _save_exact16(exact16_out, raw_dn_lin, WL)

    # Record config to meta_out.
    meta_out = dict(meta)
    meta_out.setdefault("denoise", {})
    meta_out["denoise"].update({
        "method": used_backend,
        "strength": float(strength),
        "bm3d_stage": None,
        "sigma_cap01": float(sigma_cap01),
        "sigma_floor_counts": float(sigma_floor_counts),
    })

    # Info for logs/regression.
    info = {
        "backend": used_backend,
        "sigma_used": sigma_used,
        "denom": denom,
        "shape": list(raw_lin.shape),
        "bayer": bayer
    }

    # Dump info JSON if we saved outputs.
    try:
        if save_plots or save_exact16:
            with open(os.path.join(out_dir, f"{stem}_BM3D_info.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "backend": used_backend,
                    "sigma_used": sigma_used,
                    "denom": denom,
                    "sigma_cap01": float(sigma_cap01),
                    "sigma_floor_counts": float(sigma_floor_counts),
                }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug("[BM3D] info json save skipped: %s", e)

    logger.info("[BM3D] backend=%s  sigma(R/Gr/Gb/B)=%s  saved_plots=%s",
                used_backend,
                {k: round(v,3) for k,v in sigma_used.items()},
                str(bool(save_plots)))

    return raw_dn_lin, meta_out, info
