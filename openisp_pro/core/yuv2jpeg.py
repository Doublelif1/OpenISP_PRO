# core/yuv2jpeg.py
# -*- coding: utf-8 -*-
"""
Manual JPEG Baseline (DCT, JFIF) encoder:
  - Input: YUV444 (8-bit) or YUV420 (8-bit)
  - Output: preserves input subsampling (444/420); optional 444→420 via downsample_444_to_420
  - DCT: AAN float FDCT (same lineage as libjpeg/FFmpeg)
  - Quantization: qt * (aanscales[row]*aanscales[col]*8) — aligned with libjpeg/FFmpeg

Pipeline:
  1) Standard/range normalization (bt601/full)
  2) Subsampling choice (444/420; optional 444→420)
  3) Padding to MCU boundaries
  4) Level shift (-128) + AAN FDCT
  5) Quantization (quality→qt) with AAN scaling
  6) ZigZag / RLE
  7) Huffman entropy coding (Annex K default tables)
  8) Write JFIF, DQT, DHT, SOF0 (variable sampling factors), SOS, image data, EOI

Deps: numpy + Python stdlib only
"""
import math
import struct
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import logging
_log = logging.getLogger("core.yuv2jpeg")
# ==============================
# Common: normalize to bt601 + full-range before encoding
# ==============================

_KRKGKB = {
    "bt601":  (0.299000, 0.587000, 0.114000),
    "bt709":  (0.212600, 0.715200, 0.072200),
    "bt2020": (0.262700, 0.678000, 0.059300),
}

def jpeg_quant_stats(Yp, Cbp, Crp, YHV, CHV, QY, QC):
    """
    Scan all 8x8 blocks once with AAN FDCT + quantization and report the max AC category
    (Baseline JPEG requires <= 10). Supports both 420/444 by iterating per-plane size.
    """
    y_h, y_v = YHV
    c_h, c_v = CHV

    def ac_max_category(img: np.ndarray, qtbl: np.ndarray, tiles_h: int, tiles_v: int) -> int:
        H, W = img.shape
        step_h = tiles_v * 8
        step_w = tiles_h * 8
        # Align to whole tiles (should already be padded)
        H_lim = (H // step_h) * step_h
        W_lim = (W // step_w) * step_w
        ac_cat_max = 0
        for my in range(0, H_lim, step_h):
            for mx in range(0, W_lim, step_w):
                for by in range(0, step_h, 8):
                    for bx in range(0, step_w, 8):
                        blk = img[my+by:my+by+8, mx+bx:mx+bx+8]
                        # Safety: fail fast if not 8×8 to catch padding issues
                        if blk.shape != (8, 8):
                            raise RuntimeError(
                                f"[JPEG/DEBUG] non-8x8 block: {blk.shape} @ my={my},mx={mx},by={by},bx={bx}, "
                                f"plane={H}x{W}, tiles={tiles_h}x{tiles_v}"
                            )
                        dctb = fdct8x8_aan(blk)
                        qblk = _quantize_aan(dctb, qtbl)
                        ac = np.abs(qblk.flatten()[1:]).astype(np.float64)
                        # Category via frexp exponent: floor(log2(x))+1; x==0 → 0
                        _, e = np.frexp(ac)
                        cat_max_here = int(e.max(initial=0))
                        if cat_max_here > ac_cat_max:
                            ac_cat_max = cat_max_here
        return ac_cat_max

    acY  = ac_max_category(Yp,  QY, y_h, y_v)
    acCb = ac_max_category(Cbp, QC, c_h, c_v)
    acCr = ac_max_category(Crp, QC, c_h, c_v)

    ac_max = max(acY, acCb, acCr)
    _log.info("[JPEG/DEBUG] AC max category: Y=%d Cb=%d Cr=%d  (baseline<=10)", acY, acCb, acCr)
    if ac_max > 10:
        _log.warning("[JPEG/DEBUG] AC category overflow: %d (>10). Check AAN scaling/quantization.", ac_max)



def _to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.uint8:
        x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return x

def _limited_to_full_8bit(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """limited-range (Y:16..235, C:16..240) → full-range (0..255)"""
    Yf  = (Y.astype(np.float32) - 16.0)  * (255.0 / 219.0)
    Uf  = (U.astype(np.float32) - 128.0) * (255.0 / 224.0) + 128.0
    Vf  = (V.astype(np.float32) - 128.0) * (255.0 / 224.0) + 128.0
    return _to_uint8(Yf), _to_uint8(Uf), _to_uint8(Vf)

def _ycbcr_to_rgb_full(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray, standard: str) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Full-range Y'CbCr → RGB using the given standard (bt601/709/2020)."""
    Kr, Kg, Kb = _KRKGKB[standard]
    Y  = Y.astype(np.float32)
    Cb = Cb.astype(np.float32) - 128.0
    Cr = Cr.astype(np.float32) - 128.0
    R = Y + 2.0*(1.0-Kr)*Cr
    G = Y - 2.0*Kb*(1.0-Kb)/Kg * Cb - 2.0*Kr*(1.0-Kr)/Kg * Cr
    B = Y + 2.0*(1.0-Kb)*Cb
    return _to_uint8(R), _to_uint8(G), _to_uint8(B)

def _rgb_to_ycbcr_bt601_full(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """RGB(0..255) → bt601 full-range Y'CbCr444."""
    R = R.astype(np.float32); G = G.astype(np.float32); B = B.astype(np.float32)
    Y  =  0.29900*R + 0.58700*G + 0.11400*B
    Cb = -0.168736*R - 0.331264*G + 0.500000*B + 128.0
    Cr =  0.500000*R - 0.418688*G - 0.081312*B + 128.0
    return _to_uint8(Y), _to_uint8(Cb), _to_uint8(Cr)

def adapt_to_bt601_full(Y: np.ndarray, U: np.ndarray, V: np.ndarray, meta: Dict,
                        adapt_non_bt601: bool = True) -> Tuple[np.ndarray,np.ndarray,np.ndarray,Dict,Dict]:
    """
    Normalize input to bt601 + full-range:
      - If meta['csc_range']=="limited", expand to full.
      - If meta['csc_standard'] is not bt601 and adapt_non_bt601=True:
        convert YUV→RGB with true standard, then RGB→bt601 YUV.
      - Otherwise raise.
    Returns: (Y,U,V) adapted, updated meta, and a side-effect report.
    """
    std = str(meta.get("csc_standard", "bt601")).lower()
    rng = str(meta.get("csc_range", "full")).lower()
    report = {"in_standard": std, "in_range": rng, "ops": []}

    Y, U, V = _to_uint8(Y), _to_uint8(U), _to_uint8(V)

    if rng == "limited":
        Y, U, V = _limited_to_full_8bit(Y, U, V)
        report["ops"].append("limited→full")

    if std != "bt601":
        if not adapt_non_bt601:
            raise ValueError(f"[yuv2jpeg] expects bt601 only, got {std}")
        if std not in _KRKGKB:
            raise ValueError(f"[yuv2jpeg] unknown standard: {std}")
        R, G, B = _ycbcr_to_rgb_full(Y, U, V, std)
        Y, U, V = _rgb_to_ycbcr_bt601_full(R, G, B)
        report["ops"].append(f"{std}→bt601 via RGB")

    meta2 = dict(meta)
    meta2["csc_standard"] = "bt601"
    meta2["csc_range"]    = "full"
    return Y, U, V, meta2, report

# ==============================
# Subsampling & MCU alignment
# ==============================

def _crop_even(img: np.ndarray, even_h: bool, even_w: bool) -> np.ndarray:
    h, w = img.shape[:2]
    hh = h - (h % 2) if even_h else h
    ww = w - (w % 2) if even_w else w
    return img[:hh, :ww]

def _downsample_420_fancy(U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    444→420 downsample using a libjpeg-like separable filter [1 3 3 1]/8:
    horizontal filter + decimate, then vertical filter + decimate.
    Edge-repeat padding keeps vectorized slices consistent.
    """
    def filt1d_h(img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        # pad: left+1, right+2
        a = np.pad(img, ((0, 0), (1, 2)), mode="edge").astype(np.float32)
        # 4-tap [1 3 3 1]/8; output (h, w)
        return (a[:, 0:w] + 3.0*a[:, 1:w+1] + 3.0*a[:, 2:w+2] + a[:, 3:w+3]) * (1.0/8.0)

    def filt1d_v(img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        # pad: top+1, bottom+2
        a = np.pad(img, ((1, 2), (0, 0)), mode="edge").astype(np.float32)
        return (a[0:h, :] + 3.0*a[1:h+1, :] + 3.0*a[2:h+2, :] + a[3:h+3, :]) * (1.0/8.0)

    def down(img: np.ndarray) -> np.ndarray:
        # horizontal filter → decimate; then vertical filter → decimate
        hpass = filt1d_h(img)[:, ::2]
        vpass = filt1d_v(hpass)[::2, :]
        return np.clip(np.rint(vpass), 0, 255).astype(np.uint8)

    return down(U), down(V)


def _detect_input_subsampling(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> str:
    """Return '444' or '420' based on plane sizes."""
    if U.shape == Y.shape and V.shape == Y.shape:
        return "444"
    H, W = Y.shape
    if U.shape == (H // 2, W // 2) and V.shape == (H // 2, W // 2):
        return "420"
    raise ValueError(f"[yuv2jpeg] unsupported shapes: Y={Y.shape}, U={U.shape}, V={V.shape} (444/420 only)")

def _pad_to_mcu_generic(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray,
                        YHV: Tuple[int,int], CHV: Tuple[int,int]
                       ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,int,int,int,int]:
    """
    Generic MCU padding:
      - YHV/CHV: (H_samp, V_samp), e.g. 420: YHV=(2,2), CHV=(1,1); 444: (1,1)
      - MCU size (pixels) = (8*max_h, 8*max_v)
      - Pad Y/Cb/Cr with edge-repeat to target sizes
    Returns: Yp, Cbp, Crp, Wm, Hm, MCU_W, MCU_H
    """
    H, W = Y.shape
    max_h = max(YHV[0], CHV[0])
    max_v = max(YHV[1], CHV[1])
    MCU_W = 8 * max_h
    MCU_H = 8 * max_v

    Wm = (W + MCU_W - 1) // MCU_W * MCU_W
    Hm = (H + MCU_H - 1) // MCU_H * MCU_H

    def pad_edge(img, target_h, target_w):
        h, w = img.shape
        out = np.empty((target_h, target_w), dtype=img.dtype)
        out[:h, :w] = img
        if w < target_w:
            out[:h, w:target_w] = img[:, w-1:w]
        if h < target_h:
            out[h:target_h, :] = out[h-1:h, :]
        return out

    # Y target
    Yp = pad_edge(Y, Hm, Wm)
    # Chroma target size: scale by sampling factors relative to Y
    target_ch = Hm * CHV[1] // YHV[1]
    target_cw = Wm * CHV[0] // YHV[0]
    Cbp = pad_edge(Cb, target_ch, target_cw)
    Crp = pad_edge(Cr, target_ch, target_cw)
    return Yp, Cbp, Crp, Wm, Hm, MCU_W, MCU_H

# ==============================
# AAN FDCT / quant / ZigZag / entropy coding
# ==============================

# AAN constants (aligned with libjpeg/FFmpeg)
_COS_4   = 0.7071067811865476  # sqrt(1/2)
_AAN_M0  = 0.382683433
_AAN_M1  = 0.541196100
_AAN_M2  = 1.306562965
_AAN_M3  = 0.707106781

def fdct8x8_aan(block: np.ndarray) -> np.ndarray:
    """
    AAN float FDCT (equivalent to IJG/FFmpeg jfdctflt.c).
    Input: 8×8 int16 (already level-shifted)
    Output: 8×8 float32 (AAN-scaled domain)
    """
    d = block.astype(np.float32).copy()

    # Pass 1: rows
    for i in range(8):
        d0 = d[i,0] + d[i,7];  d7 = d[i,0] - d[i,7]
        d1 = d[i,1] + d[i,6];  d6 = d[i,1] - d[i,6]
        d2 = d[i,2] + d[i,5];  d5 = d[i,2] - d[i,5]
        d3 = d[i,3] + d[i,4];  d4 = d[i,3] - d[i,4]

        tmp0 = d0 + d3
        tmp3 = d0 - d3
        tmp1 = d1 + d2
        tmp2 = d1 - d2

        d[i,0] = tmp0 + tmp1
        d[i,4] = tmp0 - tmp1

        z1 = (tmp2 + tmp3) * _COS_4
        d[i,2] = tmp3 + z1
        d[i,6] = tmp3 - z1

        tmp10 = d4 + d5
        tmp11 = d5 + d6
        tmp12 = d6 + d7

        z5 = (tmp10 - tmp12) * _AAN_M0
        z2 = _AAN_M1 * tmp10 + z5
        z4 = _AAN_M2 * tmp12 + z5
        z3 = tmp11 * _AAN_M3

        z11 = d7 + z3
        z13 = d7 - z3

        d[i,5] = z13 + z2
        d[i,3] = z13 - z2
        d[i,1] = z11 + z4
        d[i,7] = z11 - z4

    # Pass 2: cols
    for j in range(8):
        d0 = d[0,j] + d[7,j];  d7 = d[0,j] - d[7,j]
        d1 = d[1,j] + d[6,j];  d6 = d[1,j] - d[6,j]
        d2 = d[2,j] + d[5,j];  d5 = d[2,j] - d[5,j]
        d3 = d[3,j] + d[4,j];  d4 = d[3,j] - d[4,j]

        tmp0 = d0 + d3
        tmp3 = d0 - d3
        tmp1 = d1 + d2
        tmp2 = d1 - d2

        d[0,j] = tmp0 + tmp1
        d[4,j] = tmp0 - tmp1

        z1 = (tmp2 + tmp3) * _COS_4
        d[2,j] = tmp3 + z1
        d[6,j] = tmp3 - z1

        tmp10 = d4 + d5
        tmp11 = d5 + d6
        tmp12 = d6 + d7

        z5 = (tmp10 - tmp12) * _AAN_M0
        z2 = _AAN_M1 * tmp10 + z5
        z4 = _AAN_M2 * tmp12 + z5
        z3 = tmp11 * _AAN_M3

        z11 = d7 + z3
        z13 = d7 - z3

        d[5,j] = z13 + z2
        d[3,j] = z13 - z2
        d[1,j] = z11 + z4
        d[7,j] = z11 - z4

    return d

def _selfcheck_zigzag():
    # Build a 0..63 matrix and verify first 10 ZigZag indices
    m = np.arange(64, dtype=np.int32).reshape(8,8)
    seq = m.flatten()[ZIGZAG]
    # Expected start: 0,1,8,16,9,2,3,10,17,24
    assert list(seq[:10]) == [0,1,8,16,9,2,3,10,17,24], \
        f"ZigZag table mismatch, head={list(seq[:10])}"

ZIGZAG = np.array([
  0, 1, 8,16, 9, 2, 3,10,
 17,24,32,25,18,11, 4, 5,
 12,19,26,33,40,48,41,34,
 27,20,13, 6, 7,14,21,28,
 35,42,49,56,57,50,43,36,
 29,22,15,23,30,37,44,51,
 58,59,52,45,38,31,39,46,
 53,60,61,54,47,55,62,63
], dtype=np.int32)


# Annex K base quant tables (natural order)
_QT_LUMA = np.array([
    16,11,10,16,24,40,51,61,
    12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77,
    24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101,
    72,92,95,98,112,100,103,99
], dtype=np.int32)

_QT_CHROMA = np.array([
    17,18,24,47,99,99,99,99,
    18,21,26,66,99,99,99,99,
    24,26,56,99,99,99,99,99,
    47,66,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99
], dtype=np.int32)

def _build_qtable(base: np.ndarray, quality: int) -> np.ndarray:
    """Build encoder-side quant table from quality (natural order)."""
    q = max(1, min(quality, 100))
    scale = 5000 // q if q < 50 else 200 - 2*q
    Q = ((base.astype(np.int32) * scale + 50) // 100).clip(1, 255).astype(np.uint8)
    return Q

# AAN scaling table (matches libjpeg/FFmpeg)
_AAN_1D = np.array([1.0, 1.387039845, 1.306562965, 1.175875602,
                    1.0, 0.785694958, 0.541196100, 0.275899379], dtype=np.float32)
_AAN_2D = (_AAN_1D[:, None] * _AAN_1D[None, :]) * 8.0   # note: includes ×8



def _jrnd(x: np.ndarray) -> np.ndarray:
    # libjpeg-style round-half-away-from-zero
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5))


def _quantize_aan(dct_block, qtbl):
    qt = qtbl.reshape(8,8).astype(np.float32) * _AAN_2D
    return _jrnd(dct_block / qt).astype(np.int32)  # round-half-away-from-zero



# ====== Standard Huffman tables (Annex K, IJG-compatible) ======
# DC Luma/Chroma, AC Luma/Chroma (unchanged)
STD_DC_L_NRCODES = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
STD_DC_L_VALUES  = [0,1,2,3,4,5,6,7,8,9,10,11]
STD_DC_C_NRCODES = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
STD_DC_C_VALUES  = [0,1,2,3,4,5,6,7,8,9,10,11]
STD_AC_L_NRCODES = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d]
STD_AC_L_VALUES  = [
  0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,
  0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,
  0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,
  0x26,0x27,0x28,0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,
  0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,
  0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,
  0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,
  0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
  0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,
  0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,
  0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa
]
STD_DC_C_VALUES
STD_AC_C_NRCODES = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]
STD_AC_C_VALUES  = [
  0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,
  0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,
  0x52,0xf0,0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,
  0x19,0x1a,0x26,0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,
  0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,
  0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,
  0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,
  0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,
  0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,
  0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,
  0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa
]

def _build_huffcode(bits: List[int], vals: List[int]):
    if len(bits) == 17:
        bits = bits[:16]
    elif len(bits) != 16:
        raise ValueError(f"Huffman bits length must be 16 or 17, got {len(bits)}")
    total = sum(bits)
    if total != len(vals):
        raise ValueError(f"Sum(bits)({total}) != number of values({len(vals)})")
    huff = {}
    code = 0
    k = 0
    for i in range(16):
        for _ in range(bits[i]):
            huff[vals[k]] = (code, i+1)
            code += 1
            k += 1
        code <<= 1
    return huff

H_DC_L = _build_huffcode(STD_DC_L_NRCODES, STD_DC_L_VALUES)
H_DC_C = _build_huffcode(STD_DC_C_NRCODES, STD_DC_C_VALUES)
H_AC_L = _build_huffcode(STD_AC_L_NRCODES, STD_AC_L_VALUES)
H_AC_C = _build_huffcode(STD_AC_C_NRCODES, STD_AC_C_VALUES)

assert 0x00 in H_AC_L
assert 0xF0 in H_AC_L
for _cat in range(12):
    assert _cat in H_DC_L
    assert _cat in H_DC_C

# ========= Bitstream writer =========

class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.bits = 0

    def write(self, code: int, nbits: int):
        for i in range(nbits - 1, -1, -1):
            self.acc = (self.acc << 1) | ((code >> i) & 1)
            self.bits += 1
            if self.bits == 8:
                b = self.acc & 0xFF
                self.buf.append(b)
                if b == 0xFF:
                    self.buf.append(0x00)
                self.acc = 0
                self.bits = 0

    def flush(self):
        if self.bits > 0:
            pad = (1 << (8 - self.bits)) - 1  # pad with 1s on the tail
            self.acc = (self.acc << (8 - self.bits)) | pad
            b = self.acc & 0xFF
            self.buf.append(b)
            if b == 0xFF:
                self.buf.append(0x00)
            self.acc = 0
            self.bits = 0

def _category_amp(val: int) -> Tuple[int,int,int]:
    if val == 0:
        return 0, 0, 0
    abs_v = abs(val)
    cat = abs_v.bit_length()
    if val < 0:
        amp = ((1 << cat) - 1) + val  # negative amplitude in JPEG coding
    else:
        amp = val
    return cat, amp, cat

def _encode_block(block_q: np.ndarray, prev_dc: int, bw: BitWriter, dc_huff: Dict, ac_huff: Dict) -> int:
    zz = block_q.flatten()[ZIGZAG]
    dc = int(zz[0])
    diff = dc - prev_dc
    size, amp, nbits = _category_amp(diff)
    code, clen = dc_huff[size]
    bw.write(code, clen)
    if nbits:
        bw.write(amp, nbits)
    run = 0
    for k in range(1, 64):
        ac = int(zz[k])
        if ac == 0:
            run += 1
            if run == 16:
                code, clen = ac_huff[0xF0]  # ZRL
                bw.write(code, clen)
                run = 0
        else:
            size, amp, nbits = _category_amp(ac)
            sym = (run << 4) | size
            code_clen = ac_huff.get(sym)
            if code_clen is None:
                _run  = (sym >> 4) & 0xF
                _size = sym & 0xF
                raise ValueError(f"AC Huffman missing symbol: RS=0x{sym:02X} (run={_run}, size={_size})")
            code, clen = code_clen
            bw.write(code, clen)
            bw.write(amp, nbits)
            run = 0
    if run > 0:
        code, clen = ac_huff[0x00]  # EOB
        bw.write(code, clen)
    return dc

# ==============================
# JPEG markers (JFIF Baseline)
# ==============================

def _be16(x): return struct.pack(">H", x)
def _marker(m): return b"\xFF" + bytes([m])

def _write_APP0_JFIF() -> bytes:
    out = bytearray()
    out += _marker(0xE0)
    body = bytearray()
    body += b"JFIF\x00"
    body += b"\x01\x02"
    body += b"\x00"          # Units: 0
    body += b"\x00\x01"      # Xdensity
    body += b"\x00\x01"      # Ydensity
    body += b"\x00\x00"      # No thumbnail
    out += _be16(len(body)+2)
    out += body
    return bytes(out)

def _write_DQT(qt: np.ndarray, tid: int) -> bytes:
    out = bytearray()
    out += _marker(0xDB)
    body = bytearray()
    body += bytes([0x00 | (tid & 0x0F)])  # 8-bit precision, table id
    body += qt.reshape(64)[ZIGZAG].astype(np.uint8).tobytes()
    out += _be16(len(body)+2)
    out += body
    return bytes(out)

def _write_SOF0(width: int, height: int, YHV: Tuple[int,int], CHV: Tuple[int,int]) -> bytes:
    y_h, y_v = YHV
    c_h, c_v = CHV
    out = bytearray()
    out += _marker(0xC0)
    body = bytearray()
    body += b"\x08" + _be16(height) + _be16(width)
    body += b"\x03"
    body += b"\x01" + bytes([((y_h & 0xF) << 4) | (y_v & 0xF)]) + b"\x00"  # Y, QT=0
    body += b"\x02" + bytes([((c_h & 0xF) << 4) | (c_v & 0xF)]) + b"\x01"  # Cb, QT=1
    body += b"\x03" + bytes([((c_h & 0xF) << 4) | (c_v & 0xF)]) + b"\x01"  # Cr, QT=1
    out += _be16(len(body)+2)
    out += body
    return bytes(out)

def _write_DHT(bits: List[int], vals: List[int], table_class: int, table_id: int) -> bytes:
    if len(bits) == 17:
        bits = bits[:16]
    total = sum(bits)
    if total != len(vals):
        raise ValueError(f"DHT mismatch: sum(bits)={total} vs values={len(vals)}")
    out = bytearray()
    out += _marker(0xC4)
    body = bytearray()
    body += bytes([(table_class<<4) | (table_id & 0x0F)])
    body += bytes(bits)
    body += bytes(vals)
    out += _be16(len(body)+2)
    out += body
    return bytes(out)

def _write_SOS() -> bytes:
    out = bytearray()
    out += _marker(0xDA)
    body = bytearray()
    body += b"\x03"
    body += b"\x01" + b"\x00"  # Y: DC=0, AC=0
    body += b"\x02" + b"\x11"  # Cb: DC=1, AC=1
    body += b"\x03" + b"\x11"  # Cr: DC=1, AC=1
    body += b"\x00\x3F\x00"
    out += _be16(len(body)+2)
    out += body
    return bytes(out)

# ==============================
# Main encoder
# ==============================

def run_yuv2jpeg_manual(
    Y_in: np.ndarray, U_in: np.ndarray, V_in: np.ndarray, meta_in: Dict,
    out_dir: str, filename_stem: str = "final",
    quality: int = 60,
    adapt_non_bt601: bool = True,
    downsample_444_to_420: bool = False,

    use_luma_q_for_chroma_when_444: bool = True,
    chroma_q_scale_when_444: float = 1.00,   # small tweak 0.9~1.2 if needed
) -> Dict:
    """
    Manual JPEG Baseline encoder (supports 4:4:4 and 4:2:0, JFIF).
    Returns: {"saved_path": <path>, "report": {...}, "meta_out": {...}}
    """
    _selfcheck_zigzag()
    # 0) Normalize to bt601 full-range
    Y0, U0, V0, meta2, report = adapt_to_bt601_full(Y_in, U_in, V_in, meta_in, adapt_non_bt601)

    # 1) Detect input subsampling
    in_ss = _detect_input_subsampling(Y0, U0, V0)

    # 2) Choose subsampling
    if in_ss == "444":
        if downsample_444_to_420:
            Y = _crop_even(Y0, True, True)
            U = _crop_even(U0, True, True)
            V = _crop_even(V0, True, True)
            Cb, Cr = _downsample_420_fancy(U, V)
            assert Cb.shape == (Y.shape[0] // 2, Y.shape[1] // 2) and Cr.shape == Cb.shape, \
                f"420 size mismatch: Y={Y.shape}, Cb={Cb.shape}, Cr={Cr.shape}"
            out_ss = "420";  YHV, CHV = (2,2), (1,1)
        else:
            Y, Cb, Cr = Y0, U0, V0
            out_ss = "444";  YHV, CHV = (1,1), (1,1)
    else:  # input is 420
        Y, Cb, Cr = Y0, U0, V0
        out_ss = "420";    YHV, CHV = (2,2), (1,1)

    # Original (may be cropped to even)
    H, W = Y.shape

    # 3) MCU padding
    Yp, Cbp, Crp, Wm, Hm, MCU_W, MCU_H = _pad_to_mcu_generic(Y, Cb, Cr, YHV, CHV)

    # 4) Quant tables
    QY = _build_qtable(_QT_LUMA,   quality)
    QC = _build_qtable(_QT_CHROMA, quality)

    # For 4:4:4, prefer using luma table for chroma to avoid Annex-K chroma being too soft
    if out_ss == "444" and use_luma_q_for_chroma_when_444:
        QC = np.clip(np.rint(QY.astype(np.float32) * float(chroma_q_scale_when_444)),
                     1, 255).astype(np.uint8)
    qt_mode = ("chroma=luma*%.2f" % chroma_q_scale_when_444) \
        if (out_ss == "444" and use_luma_q_for_chroma_when_444) else "annexK"
    # Debug: one offline DCT+quant pass to check baseline category limits
    jpeg_quant_stats(Yp, Cbp, Crp, YHV, CHV, QY, QC)

    # 5) Headers
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{filename_stem}.jpg"
    jpg = bytearray()
    jpg += _marker(0xD8)                              # SOI
    jpg += _write_APP0_JFIF()                         # APP0 (JFIF)
    jpg += _write_DQT(QY, 0)                          # DQT(0)
    jpg += _write_DQT(QC, 1)                          # DQT(1)
    jpg += _write_SOF0(W, H, YHV, CHV)                # SOF0 (sampling factors)
    jpg += _write_DHT(STD_DC_L_NRCODES, STD_DC_L_VALUES, table_class=0, table_id=0)
    jpg += _write_DHT(STD_AC_L_NRCODES, STD_AC_L_VALUES, table_class=1, table_id=0)
    jpg += _write_DHT(STD_DC_C_NRCODES, STD_DC_C_VALUES, table_class=0, table_id=1)
    jpg += _write_DHT(STD_AC_C_NRCODES, STD_AC_C_VALUES, table_class=1, table_id=1)
    jpg += _write_SOS()                               # SOS
    logging.getLogger("core.yuv2jpeg").info(
        "[JPEG/DEBUG] QT(Y) head=%s ... QT(C) head=%s ... mode=%s",
        ",".join(map(str, (QY.flatten()[:8]))),
        ",".join(map(str, (QC.flatten()[:8]))),
        qt_mode
    )

    # 6) Scan MCUs & entropy coding (AAN FDCT + AAN quant)
    bw = BitWriter()
    prev_dc_Y  = 0
    prev_dc_Cb = 0
    prev_dc_Cr = 0

    # level shift (-128)
    Yls  = (Yp.astype(np.int16)  - 128).astype(np.int16)
    Cbls = (Cbp.astype(np.int16) - 128).astype(np.int16)
    Crls = (Crp.astype(np.int16) - 128).astype(np.int16)

    y_h, y_v = YHV
    c_h, c_v = CHV

    for my in range(0, Hm, MCU_H):
        for mx in range(0, Wm, MCU_W):
            # Y: y_v * y_h blocks of 8×8
            for by in range(0, y_v*8, 8):
                for bx in range(0, y_h*8, 8):
                    blk = Yls[my+by:my+by+8, mx+bx:mx+bx+8]
                    dctb = fdct8x8_aan(blk)
                    qblk = _quantize_aan(dctb, QY)
                    prev_dc_Y = _encode_block(qblk, prev_dc_Y, bw, H_DC_L, H_AC_L)

            # Chroma MCU origin in chroma grid
            c_my = (my // MCU_H) * (c_v*8)
            c_mx = (mx // MCU_W) * (c_h*8)

            for by in range(0, c_v*8, 8):
                for bx in range(0, c_h*8, 8):
                    blk = Cbls[c_my+by:c_my+by+8, c_mx+bx:c_mx+bx+8]
                    dctb = fdct8x8_aan(blk)
                    qblk = _quantize_aan(dctb, QC)
                    prev_dc_Cb = _encode_block(qblk, prev_dc_Cb, bw, H_DC_C, H_AC_C)

            for by in range(0, c_v*8, 8):
                for bx in range(0, c_h*8, 8):
                    blk = Crls[c_my+by:c_my+by+8, c_mx+bx:c_mx+bx+8]
                    dctb = fdct8x8_aan(blk)
                    qblk = _quantize_aan(dctb, QC)
                    prev_dc_Cr = _encode_block(qblk, prev_dc_Cr, bw, H_DC_C, H_AC_C)

    bw.flush()
    jpg += bw.buf
    jpg += _marker(0xD9)  # EOI

    with open(out_path, "wb") as f:
        f.write(jpg)

    report.update({
        "in_subsampling":  in_ss,
        "out_subsampling": out_ss,
        "width": int(W), "height": int(H),
        "width_padded": int(Wm), "height_padded": int(Hm),
        "quality": int(quality),
        "qtY_minmax": (int(QY.min()), int(QY.max())),
        "qtC_minmax": (int(QC.min()), int(QC.max())),
    "qt_mode": (
        "chroma=luma*%.2f" % chroma_q_scale_when_444
        if (out_ss=="444" and use_luma_q_for_chroma_when_444)
        else "annexK"
    )
    })
    return {"saved_path": str(out_path), "report": report, "meta_out": meta2}
