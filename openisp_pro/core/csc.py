# -*- coding: utf-8 -*-
"""
CSC (Color Space Conversion)
Convert R'G'B' (uint8 or float) to Y'CbCr 4:4:4 (uint8, planar).

Notes:
- Standards: BT.601 / BT.709 / BT.2020 (default: bt601; choose via run_csc_rgb_to_yuv444.standard).
- Ranges: full / limited (default: full).
- Float inputs auto-handle white_level to avoid treating [0,1] as RAW and darkening.
- Optional outputs: Y/Cb/Cr PNGs, triptych (Y|Cb|Cr), a debug RGB stack of (Y,Cb,Cr), and planar 444 .yuv.

Terminology:
- “YUV” here refers to digital Y'CbCr (Cb/Cr centered at 128).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ===================== Basics =====================

# Full-range 8-bit coefficients (R'G'B' → Y'CbCr).
# These linearly form Y' from R'G'B', and Cb/Cr by adding a 128 bias.
_COEFFS = {
    "bt601": {
        "Y":  (0.299000,  0.587000,  0.114000),
        "Cb": (-0.168736, -0.331264, 0.500000),
        "Cr": (0.500000,  -0.418688, -0.081312),
    },
    "bt709": {
        "Y":  (0.212600,  0.715200,  0.072200),
        "Cb": (-0.114572, -0.385428, 0.500000),
        "Cr": (0.500000,  -0.454153, -0.045847),
    },
    "bt2020": {
        "Y":  (0.262700,  0.678000,  0.059300),
        "Cb": (-0.139630, -0.360370, 0.500000),
        "Cr": (0.500000,  -0.459786, -0.040214),
    },
}


def _quantize_u8(x: np.ndarray) -> np.ndarray:
    """Round and clip to uint8 (0..255)."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _to_uint8_rgb(rgb_in: np.ndarray, meta: Dict[str, Any], clamp: bool = True) -> np.ndarray:
    """Normalize any-depth/float RGB to uint8."""
    if rgb_in is None:
        raise ValueError("_to_uint8_rgb: rgb_in is None")
    if rgb_in.ndim != 3 or rgb_in.shape[2] != 3:
        raise ValueError(f"_to_uint8_rgb: expect HxWx3, got {rgb_in.shape}")

    if rgb_in.dtype == np.uint8:
        return rgb_in.copy()

    wl = float(meta.get("white_level", 255.0))
    if not np.isfinite(wl) or wl <= 0:
        if np.issubdtype(rgb_in.dtype, np.integer):
            wl = float(np.iinfo(rgb_in.dtype).max)
        else:
            wl = 1.0
        logger.warning("[CSC] meta.white_level missing/invalid, fallback to %.1f", wl)

    # If float input looks like [0,1], treat WL as 1.0 to avoid unintended darkening.
    if np.issubdtype(rgb_in.dtype, np.floating):
        try:
            fmax = float(np.nanmax(rgb_in))
        except Exception:
            fmax = 0.0
        if fmax <= 1.5 and wl > 4:
            logger.warning("[CSC] float input max≈%.4f but WL=%.1f; use WL=1.0 to avoid darkening", fmax, wl)
            wl = 1.0

    x01 = rgb_in.astype(np.float32) / max(wl, 1e-6)
    if clamp:
        x01 = np.clip(x01, 0.0, 1.0)
    x255 = _quantize_u8(x01 * 255.0)
    return x255


def _rgb_to_ycbcr444_u8(rgb_u8: np.ndarray, standard: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R'G'B' (uint8) → Y'CbCr 4:4:4 (uint8, planar) per standard (bt601/bt709/bt2020)."""
    std = standard.lower()
    if std not in _COEFFS:
        raise ValueError(f"unsupported standard: {standard}")

    (kr, kg, kb) = _COEFFS[std]["Y"]
    (ucr, ucg, ucb) = _COEFFS[std]["Cb"]
    (vcr, vcg, vcb) = _COEFFS[std]["Cr"]

    R = rgb_u8[..., 0].astype(np.float32)
    G = rgb_u8[..., 1].astype(np.float32)
    B = rgb_u8[..., 2].astype(np.float32)

    Y  = kr*R + kg*G + kb*B
    Cb = ucr*R + ucg*G + ucb*B + 128.0
    Cr = vcr*R + vcg*G + vcb*B + 128.0

    return _quantize_u8(Y), _quantize_u8(Cb), _quantize_u8(Cr)


def _apply_range_mode(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, range_mode: str
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map full-range (0..255, neutral 128) to target range (full/limited)."""
    m = (range_mode or "full").lower()
    if m == "full":
        return Y, Cb, Cr
    if m == "limited":
        Yf  = 16.0  + (219.0/255.0) * Y
        Cbf = 128.0 + (224.0/255.0) * (Cb - 128.0)
        Crf = 128.0 + (224.0/255.0) * (Cr - 128.0)
        return _quantize_u8(Yf), _quantize_u8(Cbf), _quantize_u8(Crf)
    raise ValueError(f"unsupported range_mode: {range_mode}")

# ===================== I/O =====================

def _save_png(img8: np.ndarray, path: Path) -> Path:
    Image.fromarray(img8).save(Path(path))
    return Path(path)


def _write_planar_yuv444(outfile: Path, Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> Path:
    """Write planar 4:4:4 .yuv as Y, U(Cb), V(Cr) in order."""
    outfile = Path(outfile)
    h, w = Y.shape
    assert U.shape == (h, w) and V.shape == (h, w), "Y/U/V shapes must match"
    Yc = np.ascontiguousarray(Y)
    Uc = np.ascontiguousarray(U)
    Vc = np.ascontiguousarray(V)
    with open(outfile, "wb") as f:
        f.write(Yc.tobytes(order="C"))
        f.write(Uc.tobytes(order="C"))
        f.write(Vc.tobytes(order="C"))
    return outfile

# ========== Optional: composites for debugging/visualization ==========

def _make_triptych(Y: np.ndarray, U: np.ndarray, V: np.ndarray, gap: int = 8, bg: int = 0) -> np.ndarray:
    """Concatenate Y, U, V (uint8) side by side into a single L-mode image."""
    h, w = Y.shape
    canvas = np.full((h, 3 * w + 2 * gap), int(bg), dtype=np.uint8)
    canvas[:, 0:w] = Y
    canvas[:, w + gap:w + gap + w] = U
    canvas[:, 2 * w + 2 * gap:2 * w + 2 * gap + w] = V
    return canvas


def _save_triptych(Y: np.ndarray, U: np.ndarray, V: np.ndarray, path: Path) -> Path:
    img = _make_triptych(Y, U, V, gap=8, bg=0)
    Image.fromarray(img, mode='L').save(path)
    return Path(path)


def _save_yuv_as_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray, path: Path) -> Path:
    """Stack (Y,U,V) into (R,G,B) for a quick debug view (U=Cb, V=Cr)."""
    rgb = np.stack([Y, U, V], axis=-1)
    Image.fromarray(rgb, mode='RGB').save(path)
    return Path(path)

# ===================== Entry Point =====================

def run_csc_rgb_to_yuv444(
    rgb_in: np.ndarray,
    meta_in: Dict[str, Any],
    out_dir: str,
    *,
    standard: str = "bt601",          # "bt601" / "bt709" / "bt2020"
    range_mode: str = "full",         # "full" or "limited"
    clamp: bool = True,
    save_raw: bool = True,
    save_png: bool = True,
    save_composite: bool = True,      # also save triptych / stacked RGB debug
    filename_stem: str = "yuv444",
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """
    Convert R'G'B' to Y'CbCr 4:4:4 (uint8, planar).

    Returns:
        (Y, Cb, Cr), meta_out, info
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Normalize to 8-bit RGB first.
    rgb8 = _to_uint8_rgb(rgb_in, meta_in, clamp=clamp)

    std = standard.lower()
    if std not in _COEFFS:
        logger.warning("[CSC] unsupported standard %s, fallback to bt601", standard)
        std = "bt601"

    # Forward transform (full-range coefficients).
    Y, Cb, Cr = _rgb_to_ycbcr444_u8(rgb8, std)

    # Map to requested range.
    Y, Cb, Cr = _apply_range_mode(Y, Cb, Cr, range_mode)

    # Save artifacts.
    saves: Dict[str, Any] = {}
    stem = filename_stem

    if save_raw:
        raw_path = out_dir_p / f"{stem}.yuv"  # planar 444 (Y, Cb, Cr)
        _write_planar_yuv444(raw_path, Y, Cb, Cr)
        saves["yuv444_planar"] = str(raw_path)

    if save_png:
        y_path  = _save_png(Y,  out_dir_p / f"{stem}_Y.png")
        cb_path = _save_png(Cb, out_dir_p / f"{stem}_Cb.png")
        cr_path = _save_png(Cr, out_dir_p / f"{stem}_Cr.png")
        saves.update({
            "Y_png":  str(y_path),
            "Cb_png": str(cb_path),
            "Cr_png": str(cr_path),
        })

    if save_composite:
        trip_path    = _save_triptych(Y, Cb, Cr, out_dir_p / f"{stem}_triptych.png")
        yuvrgb_path  = _save_yuv_as_rgb(Y, Cb, Cr, out_dir_p / f"{stem}_ycbcr_as_rgb.png")
        saves.update({
            "triptych_png":     str(trip_path),
            "ycbcr_as_rgb_png": str(yuvrgb_path),
        })

    # Meta / info (mirrors implementation).
    meta_out = dict(meta_in)
    meta_out.update({
        "colorspace": "YCbCr",
        "yuv_format": "444",
        "range": range_mode.lower(),
        "matrix": std,
        "bitdepth": 8,
    })

    info = {
        "standard": std,
        "range": range_mode.lower(),
        "coeffs": _COEFFS[std],  # {'Y':(...), 'Cb':(...), 'Cr':(...)}
        "saved": saves,
    }

    logger.info("[CSC] RGB -> Y'CbCr 4:4:4 complete (%s, %s): saved=%s",
                std, range_mode, json.dumps(saves, ensure_ascii=False))
    return (Y, Cb, Cr), meta_out, info
