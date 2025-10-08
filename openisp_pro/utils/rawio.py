# rawio.py
import rawpy
import numpy as np
import math
import os

# --- Bayer name parsing ---
def _bayer_from_rawpy(raw) -> str:
    pat = np.array(raw.raw_pattern, dtype=int)  # (2,2) CFA index grid
    desc = raw.color_desc
    if isinstance(desc, bytes):
        desc = desc.decode("ascii", errors="ignore")
    desc = desc.upper()

    def ch_of(idx: int) -> str:
        if 0 <= idx < len(desc):
            ch = desc[idx]
            return ch if ch in ("R","G","B") else "G"
        return "G"

    tpl = (ch_of(pat[0,0]), ch_of(pat[0,1]), ch_of(pat[1,0]), ch_of(pat[1,1]))
    mapping = {
        "RGGB": ("R","G","G","B"),
        "GRBG": ("G","R","B","G"),
        "GBRG": ("G","B","R","G"),
        "BGGR": ("B","G","G","R"),
    }
    for name, m in mapping.items():
        if tpl == m:
            return name
    if tpl.count("R")==1 and tpl.count("B")==1 and tpl.count("G")==2:
        if   tpl == ("R","G","G","B"): return "RGGB"
        if   tpl == ("G","R","B","G"): return "GRBG"
        if   tpl == ("G","B","R","G"): return "GBRG"
        if   tpl == ("B","G","G","R"): return "BGGR"
    return "RGGB"

# 2x2 site offsets (y,x), order (R, Gr, Gb, B). Aligned with black_level.py.
_BAYER_OFFSETS = {
    "RGGB": ((0,0), (0,1), (1,0), (1,1)),
    "GRBG": ((0,1), (0,0), (1,1), (1,0)),
    "GBRG": ((1,0), (1,1), (0,0), (0,1)),
    "BGGR": ((1,1), (1,0), (0,1), (0,0)),
}

def _safe_bit_depth(raw, white_level: float) -> int:
    bd = getattr(raw, "bit_depth", None)
    if bd is not None:
        try: return int(bd)
        except: pass
    if white_level and white_level > 0:
        try:
            est = int(math.ceil(math.log2(float(white_level) + 1.0)))
            return int(np.clip(est, 8, 16))
        except:
            pass
    return 12

def _per_site_black_levels_from_tags(raw, bayer_name: str):
    """Prefer per-site black level from tags; falls back to scalar if present, else None."""
    blc = getattr(raw, "black_level_per_channel", None)
    if blc is None:
        bl_scalar = getattr(raw, "black_level", None)
        if bl_scalar is None:
            return None
        v = float(bl_scalar)
        return {"R": v, "Gr": v, "Gb": v, "B": v}

    blc = np.array(blc, dtype=np.float32).ravel()
    if blc.size == 0:
        return None

    pat = np.array(raw.raw_pattern, dtype=int)  # 2x2 channel indices
    site_bl = np.zeros((2,2), dtype=np.float32)
    max_idx = blc.size - 1
    for y in range(2):
        for x in range(2):
            idx = int(pat[y, x]); idx = max(0, min(idx, max_idx))
            site_bl[y, x] = blc[idx]

    name = (bayer_name or "RGGB").upper()
    if name not in _BAYER_OFFSETS:
        name = "RGGB"
    (yR,xR),(yGr,xGr),(yGb,xGb),(yB,xB) = _BAYER_OFFSETS[name]
    return {
        "R":  float(site_bl[yR,  xR]),
        "Gr": float(site_bl[yGr, xGr]),
        "Gb": float(site_bl[yGb, xGb]),
        "B":  float(site_bl[yB,  xB]),
    }

def save_rgb_preview_srgb(rgb_lin, out_png, black_q=0.01, white_q=0.998, gamma=2.2, exposure=1.0):
    """Quick sRGB preview from linear RGB with percentile stretch; for visualization only."""
    import imageio.v2 as iio
    import numpy as np
    x = np.asarray(rgb_lin, np.float32)
    x = np.clip(x, 0, None) * float(exposure)

    # Use approx. Rec.709 luma for a single-scale stretch (avoids per-channel banding).
    Y = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
    lo = float(np.quantile(Y, black_q)) if Y.size else 0.0
    hi = float(np.quantile(Y, white_q)) if Y.size else 1.0
    if not np.isfinite(lo): lo = 0.0
    if (not np.isfinite(hi)) or hi <= lo: hi = lo + 1.0

    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    x = np.power(x, 1.0/gamma)  # gamma to display space
    iio.imwrite(out_png, np.rint(x*255).astype(np.uint8))


# ---------- Stats helpers ----------
def _low_percentile_mean(arr, p=0.05):
    """Mean of the lowest p% after clipping the top 5% outliers; robust dark estimate."""
    arr = np.asarray(arr, np.float32)
    if arr.size == 0:
        return 0.0
    hi = np.percentile(arr, 95.0)
    arr = arr[arr <= hi]
    k = max(1, int(arr.size * p))
    part = np.partition(arr, k-1)[:k]
    return float(np.mean(part))

def _bayer_masks_full(h, w, bayer_name: str):
    """Boolean masks for full-res R/G/B positions under the given Bayer pattern."""
    (yR,xR),(yGr,xGr),(yGb,xGb),(yB,xB) = _BAYER_OFFSETS[bayer_name]
    R = np.zeros((h,w), bool); G = np.zeros((h,w), bool); B = np.zeros((h,w), bool)
    R[yR::2, xR::2]   = True
    G[yGr::2, xGr::2] = True
    G[yGb::2, xGb::2] = True
    B[yB::2, xB::2]   = True
    return R, G, B

# ---------- Optical black (OB) estimation ----------
def _estimate_black_from_ob(raw, bayer_name: str, p=0.05, margin_min_pixels=256):
    """
    Estimate per-site black from optical-black margins: take the lowest p% mean per CFA site.
    Returns None if OB region is too small.
    """
    full = raw.raw_image.copy().astype(np.float32)
    sizes = raw.sizes
    raw_h, raw_w = int(sizes.raw_height), int(sizes.raw_width)
    top, left = int(sizes.top_margin), int(sizes.left_margin)
    vis_h, vis_w = int(sizes.height), int(sizes.width)

    y0, y1 = top, top + vis_h
    x0, x1 = left, left + vis_w

    ob_mask = np.ones((raw_h, raw_w), bool)
    ob_mask[y0:y1, x0:x1] = False  # exclude visible area

    if ob_mask.sum() < margin_min_pixels:
        return None

    (yR,xR),(yGr,xGr),(yGb,xGb),(yB,xB) = _BAYER_OFFSETS[bayer_name]
    Rm = np.zeros((raw_h, raw_w), bool); Rm[yR::2, xR::2]   = True
    Grm= np.zeros((raw_h, raw_w), bool); Grm[yGr::2, xGr::2] = True
    Gbm= np.zeros((raw_h, raw_w), bool); Gbm[yGb::2, xGb::2] = True
    Bm = np.zeros((raw_h, raw_w), bool); Bm[yB::2, xB::2]   = True

    r_bl  = _low_percentile_mean(full[ob_mask & Rm],  p)
    gr_bl = _low_percentile_mean(full[ob_mask & Grm], p)
    gb_bl = _low_percentile_mean(full[ob_mask & Gbm], p)
    b_bl  = _low_percentile_mean(full[ob_mask & Bm],  p)

    return {"R": r_bl, "Gr": gr_bl, "Gb": gb_bl, "B": b_bl}

# ---------- Visible-area fallback ----------
def _estimate_black_from_visible(bayer_visible: np.ndarray, bayer_name: str, p=0.01):
    """
    Fallback per-site black when tags/OB fail: on the visible area, take the lowest p% mean per site.
    We also implicitly down-weight highlights by not using the top tail.
    """
    img = np.asarray(bayer_visible, np.float32)
    h, w = img.shape
    (yR,xR),(yGr,xGr),(yGb,xGb),(yB,xB) = _BAYER_OFFSETS[bayer_name]

    Rm  = np.zeros((h, w), bool); Rm[yR::2,  xR::2]  = True
    Grm = np.zeros((h, w), bool); Grm[yGr::2, xGr::2] = True
    Gbm = np.zeros((h, w), bool); Gbm[yGb::2, xGb::2] = True
    Bm  = np.zeros((h, w), bool); Bm[yB::2,  xB::2]  = True

    r_bl  = _low_percentile_mean(img[Rm],  p)
    gr_bl = _low_percentile_mean(img[Grm], p)
    gb_bl = _low_percentile_mean(img[Gbm], p)
    b_bl  = _low_percentile_mean(img[Bm],  p)

    return {"R": r_bl, "Gr": gr_bl, "Gb": gb_bl, "B": b_bl}

def load_dng(path: str):
    """
    Returns:
      bayer: np.float32 raw visible mosaic (counts, no normalization)
      meta: dict with:
        - bayer: "RGGB/GRBG/GBRG/BGGR"
        - black_level: scalar or {'R','Gr','Gb','B'}
        - black_level_source: 'per_site' | 'scalar' | 'ob_estimated' | 'visible_estimated' | 'none'
        - white_level: float
        - bit_depth: int
        - cfa: original 2x2 CFA pattern (for debugging)
    """
    with rawpy.imread(path) as raw:
        # Visible region only (no BLC/normalization)
        bayer = raw.raw_image_visible.copy().astype(np.float32)

        # Pattern / white level / bit depth
        bayer_name  = _bayer_from_rawpy(raw)
        white_level = float(getattr(raw, "white_level", (1 << 12) - 1))
        bit_depth   = _safe_bit_depth(raw, white_level)

        # (1) tags: prefer per-site; fall back to scalar
        bl = _per_site_black_levels_from_tags(raw, bayer_name)
        bl_source = "per_site" if isinstance(bl, dict) else ("scalar" if isinstance(bl, (int,float)) else None)

        # Treat all-zero (or near-zero) tag as invalid
        if isinstance(bl, dict) and all(abs(v) < 1e-6 for v in bl.values()):
            bl = None
            bl_source = None

        # (2) OB estimate if available
        if bl is None:
            bl_ob = _estimate_black_from_ob(raw, bayer_name, p=0.05, margin_min_pixels=256)
            if bl_ob is not None:
                bl = bl_ob
                bl_source = "ob_estimated"

        # (3) Visible-area fallback
        if bl is None:
            bl_vis = _estimate_black_from_visible(bayer, bayer_name, p=0.01)
            bl = bl_vis
            bl_source = "visible_estimated"

        meta = {
            "bayer": bayer_name,
            "black_level": bl,
            "black_level_source": bl_source,
            "white_level": white_level,
            "bit_depth": bit_depth,
            "cfa": np.array(raw.raw_pattern, dtype=int).tolist(),
        }
    return bayer, meta
