# -*- coding: utf-8 -*-
"""
Pipeline: DPC -> BLC -> BM3D -> AWB -> DEMOSAIC -> CCM -> GAMMA -> CSC -> TMC -> EE -> SAT -> CTR -> HUE -> JPEG
Adds size/dtype/memory monitoring and YUV→RGB preview helpers at each stage.
"""
import json
import sys, argparse, glob
from pathlib import Path
from typing import Optional,Dict,Tuple
import traceback
import numpy as np

from utils.rawio import load_dng
from core.blc import run_auto_blc
from core.dpc import run_dpc_on_single_raw
from core.bm3d import run_bm3d_raw
from core.awb import run_awb_raw
from core.demosaic import run_demosaic_raw, _bayer_masks
from core.ccm import run_ccm_rgb
from core.gamma import run_gamma_rgb
from core.csc import run_csc_rgb_to_yuv444
from core.tmc import run_tmc_after_csc  # TMC
from core.ee import run_edge_enhance_after_tmc
from core.sat import run_sat_after_ee
from core.contrast import run_contrast_after_sat
from core.hue import run_hue_after_contrast
from core.yuv2jpeg import run_yuv2jpeg_manual

from PIL import Image

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def monitor_size(stage_name, input_data, output_data, meta=None):
    """
    Compact stage monitor: logs shape, dtype, and memory for input/output; warns on changes and meta mismatch.
    Args:
        stage_name: name of the stage
        input_data: ndarray or tuple of ndarrays
        output_data: ndarray or tuple of ndarrays
        meta: optional metadata dict
    """

    def get_shape_info(data):
        if data is None:
            return "None"
        elif isinstance(data, tuple):
            # For planar YUV etc.
            shapes = []
            for i, channel in enumerate(data):
                if isinstance(channel, np.ndarray):
                    shapes.append(f"Ch{i}:{channel.shape}")
            return " | ".join(shapes)
        elif isinstance(data, np.ndarray):
            return str(data.shape)
        else:
            return f"Unknown type: {type(data)}"

    def get_dtype_info(data):
        if data is None:
            return "None"
        elif isinstance(data, tuple):
            dtypes = []
            for channel in data:
                if isinstance(channel, np.ndarray):
                    dtypes.append(str(channel.dtype))
            return " | ".join(dtypes)
        elif isinstance(data, np.ndarray):
            return str(data.dtype)
        else:
            return "Unknown"

    def get_memory_mb(data):
        """Return total memory usage in MB for ndarray or tuple of ndarrays."""
        if data is None:
            return 0
        elif isinstance(data, tuple):
            total_bytes = 0
            for channel in data:
                if isinstance(channel, np.ndarray):
                    total_bytes += channel.nbytes
            return total_bytes / (1024 * 1024)
        elif isinstance(data, np.ndarray):
            return data.nbytes / (1024 * 1024)
        else:
            return 0

    input_shape = get_shape_info(input_data)
    output_shape = get_shape_info(output_data)
    input_dtype = get_dtype_info(input_data)
    output_dtype = get_dtype_info(output_data)
    input_mem = get_memory_mb(input_data)
    output_mem = get_memory_mb(output_data)

    # Shape / dtype change flags
    size_changed = input_shape != output_shape
    dtype_changed = input_dtype != output_dtype

    # Basic log
    logger.info(f"[SIZE_MONITOR][{stage_name}] Input: {input_shape} ({input_dtype}) | "
                f"Output: {output_shape} ({output_dtype}) | "
                f"Memory: {input_mem:.2f}MB -> {output_mem:.2f}MB")

    # Alerts
    if size_changed:
        logger.warning(f"[SIZE_MONITOR][{stage_name}] ⚠️ Size changed: {input_shape} -> {output_shape}")
    if dtype_changed:
        logger.info(f"[SIZE_MONITOR][{stage_name}] ℹ️ Dtype changed: {input_dtype} -> {output_dtype}")

    # Cross-check against meta.w/h if provided
    if meta and isinstance(output_data, np.ndarray) and output_data.ndim >= 2:
        if "width" in meta and "height" in meta:
            expected_h, expected_w = int(meta["height"]), int(meta["width"])
            actual_h, actual_w = output_data.shape[:2]
            if expected_h != actual_h or expected_w != actual_w:
                logger.warning(f"[SIZE_MONITOR][{stage_name}] ⚠️ Meta size mismatch: "
                               f"meta=(H:{expected_h},W:{expected_w}) vs actual=(H:{actual_h},W:{actual_w})")

def yuv444_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                  standard: str = "bt601",
                  range_mode: str = "full",
                  output_dtype: str = "float32") -> np.ndarray:
    """
    Convert planar Y'CbCr 4:4:4 (8-bit) to RGB preview (full-range domain). Supports BT.601/709/2020 and full/limited.
    Returns (H, W, 3) RGB.
    """
    # Input to float32
    Y  = np.asarray(Y, dtype=np.float32)
    Cb = np.asarray(U, dtype=np.float32)
    Cr = np.asarray(V, dtype=np.float32)

    # Limited → full normalization (digital range)
    m = (range_mode or "full").lower()
    if m == "limited":
        Yf  = (Y  - 16.0)  * (255.0 / 219.0)
        Cbf = (Cb - 128.0) * (255.0 / 224.0)
        Crf = (Cr - 128.0) * (255.0 / 224.0)
        Y, Cb, Cr = Yf, Cbf + 128.0, Crf + 128.0
    elif m != "full":
        logger.warning(f"[yuv444_to_rgb] Unknown range_mode '{range_mode}', assume full")

    # Fetch Kr,Kg,Kb by standard
    std = (standard or "bt601").lower()
    KRKGKB = {
        "bt601":  (0.299000, 0.587000, 0.114000),
        "bt709":  (0.212600, 0.715200, 0.072200),
        "bt2020": (0.262700, 0.678000, 0.059300),
    }
    if std not in KRKGKB:
        logger.warning(f"[yuv444_to_rgb] Unknown standard '{standard}', using BT.601")
        std = "bt601"
    Kr, Kg, Kb = KRKGKB[std]

    # Zero-center chroma
    Cb_ = Cb - 128.0
    Cr_ = Cr - 128.0

    # Inverse transform (full-range)
    coef_RV = 2.0 * (1.0 - Kr)
    coef_BU = 2.0 * (1.0 - Kb)
    coef_GU = 2.0 * Kb * (1.0 - Kb) / Kg
    coef_GV = 2.0 * Kr * (1.0 - Kr) / Kg

    R = Y + coef_RV * Cr_
    G = Y - coef_GU * Cb_ - coef_GV * Cr_
    B = Y + coef_BU * Cb_

    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0)

    if output_dtype == "uint8":
        rgb = (rgb + 0.5).astype(np.uint8)
    else:
        rgb = rgb.astype(np.float32)
    return rgb


def save_yuv_as_rgb_preview(Y: np.ndarray, U: np.ndarray, V: np.ndarray,
                            meta: dict,
                            out_dir: Path,
                            filename: str = "preview_yuv2rgb.png",
                            standard: str = "bt601",
                            range_mode: str = "full",
                            max_side: int = 1600) -> Path:
    """
    Convert YUV444 to an RGB preview and save it; optional downscale by max_side for quick inspection.
    Returns the saved path.
    """
    try:
        # YUV → RGB (uint8)
        rgb_uint8 = yuv444_to_rgb(Y, U, V, standard=standard, range_mode=range_mode, output_dtype="uint8")

        # Resize for preview if needed
        rgb_uint8 = _resize_max_side_uint8(rgb_uint8, max_side=max_side)

        # Save
        out_path = Path(out_dir) / filename
        Image.fromarray(rgb_uint8, mode='RGB').save(out_path)

        logger.info(f"[YUV2RGB Preview] Saved: {out_path} | Shape: {rgb_uint8.shape}")
        return out_path
    except Exception as e:
        logger.error(f"[YUV2RGB Preview] Failed to save preview: {e}")
        raise


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="DNG file(s) or dir(s); glob is supported")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Centralized output dir; default creates <stem>_output beside each file")

    # ---------- DPC (updated) ----------
    ap.add_argument("--dpc-modes", type=str, default="mean,gradient,median",
                    help="DPC strategy set, comma-separated")
    ap.add_argument("--dpc-pick", type=str, default="gradient",
                    choices=["mean", "gradient", "median"],
                    help="Which strategy feeds BLC")

    ap.add_argument("--dpc-thres-mode", type=str, default="auto",
                    choices=["abs", "rel", "auto"],
                    help="Threshold mode: abs (level), rel (WL ratio), auto (noise-adaptive via MAD→σ)")

    ap.add_argument("--dpc-thres", type=float, default=0.006,
                    help="If abs: absolute level; if rel: WL ratio (e.g. 0.006=0.6%%·WL); ignored in auto")

    ap.add_argument("--dpc-auto-k", type=float, default=6.0,
                    help="For auto: thres = k * σ (σ from same-color diff noise estimate)")

    ap.add_argument("--dpc-bad-min", type=int, default=8,
                    help="Bad pixel judge ≥M/8 out-of-range; typical 6~8 (default 8, conservative)")

    ap.add_argument("--dpc-bad-num", type=int, default=0,
                    help="Demo only: inject N bad pixels (0 in production)")

    # ---------- BLC ----------
    ap.add_argument("--p-list", type=str, default=[0.008, 0.010, 0.012, 0.015, 0.020],
                    help="Candidate percentiles; list or comma-separated string")
    ap.add_argument("--scale-list", type=str, default=[1.00, 1.02, 1.05, 1.08],
                    help="Scaling factors applied to the estimated black level")
    ap.add_argument("--neg-max", type=float, default=0.002,
                    help="Over-subtraction guard: max allowed negative ratio")

    # ---------- BM3D ----------
    ap.add_argument("--bm3d-method", type=str, default="auto", choices=["auto","bm3d","nlm"],
                    help="Denoiser backend; auto prefers bm3d")
    ap.add_argument("--bm3d-strength", type=float, default=0.8,
                    help="Denoise strength multiplier")
    ap.add_argument("--bm3d-save-plots", action="store_true", default=True,
                    help="Save 8-bit previews before/after BM3D")

    # Optional skip for BM3D
    ap.add_argument("--skip-bm3d", action="store_true", default=False,
                    help="Skip BM3D and pass BLC output to AWB")

    # ---------- AWB ----------
    ap.add_argument("--awb-top-percent", type=str, default="auto",
                    help="Top%% for AWB stats (number or 'auto'/'adaptive')")
    ap.add_argument("--awb-lower-frac", type=float, default=0.01,
                    help="Lower threshold as WL fraction")
    ap.add_argument("--awb-upper-frac", type=float, default=0.98,
                    help="Upper threshold as WL fraction")
    ap.add_argument("--awb-save-plots", action="store_true", default=True,
                    help="Save AWB 8-bit previews (2x2 block downsample)")

    # ---------- CCM (after DEMOSAIC) ----------
    ap.add_argument("--ccm-color-space", type=str, default="srgb",
                    choices=["srgb", "adobe-rgb-1998"],
                    help="Target color space")
    ap.add_argument("--ccm-illuminant", type=str, default="d65",
                    choices=["d65", "d50"],
                    help="Illuminant for CCM")
    ap.add_argument("--ccm-save-plots", action="store_true", default=True,
                    help="Save CCM 8-bit preview")

    # ---------- GAMMA ----------
    ap.add_argument("--gamma", type=float, default=0.8,
                    help="Gamma exponent (>0), applied on WL-normalized data")
    ap.add_argument("--gamma-save-bitdepth", type=int, choices=[8,10,12,14,16], default=8,
                    help="Output bitdepth: 8→PNG; 10/12/14/16→16-bit TIFF container")
    ap.add_argument("--gamma-no-clamp", action="store_true", default=False,
                    help="Disable [0,1] clamp after normalization/power")
    ap.add_argument("--gamma-save-plots", action="store_true", default=True,
                    help="Save sRGB previews before/after Gamma")

    # ---------- CSC (RGB -> YCbCr 4:4:4) ----------
    ap.add_argument("--csc-enable", action="store_true", default=True,
                    help="Enable RGB→YCbCr444 export")
    ap.add_argument("--csc-standard", type=str, default="bt601",
                    choices=["bt601", "bt709", "bt2020"],
                    help="CSC matrix standard")
    ap.add_argument("--csc-filename-stem", type =str, default="yuv444",
                    help="Filename stem for CSC outputs")
    ap.add_argument("--csc-save-raw", action="store_true", default=True,
                    help="Save planar .yuv")
    ap.add_argument("--csc-save-png", action="store_true", default=True,
                    help="Save Y/Cb/Cr PNG previews")
    ap.add_argument("--csc-save-rgb-preview", action="store_true", default=True,
                    help="Save YUV→RGB preview image")

    # ---------- TMC (after CSC; adjust Y' only) ----------
    ap.add_argument("--tmc-enable", action="store_true", default=True,
                    help="Enable tone mapping on Y' after CSC")
    ap.add_argument("--tmc-curve", type=str, default="auto", choices=["auto", "aces", "hable"],
                    help="Tone-curve selection")
    ap.add_argument("--tmc-local-radius", type=int, default=8,
                    help="Local-lite radius in pixels")
    ap.add_argument("--tmc-local-detail-pow", type=float, default=1.02,
                    help="Local-lite detail exponent")
    ap.add_argument("--tmc-save-preview", action="store_true", default=True,
                    help="Save RGB preview after TMC")
    ap.add_argument("--tmc-save-rgb-preview", action="store_true", default=True,
                    help="Save YUV→RGB preview after TMC")

    # ---------- EE (after TMC; sharpen Y') ----------
    ap.add_argument("--ee-enable", action="store_true", default=True,
                    help="Enable Y' sharpening after TMC")
    ap.add_argument("--ee-amount", type=float, default=4.0,
                    help="Sharpen strength (0~1 typical)")
    ap.add_argument("--ee-radius", type=int, default=3,
                    help="Blur radius in pixels (1~2 common)")
    ap.add_argument("--ee-threshold", type=float, default=0.002,
                    help="Soft threshold (0~1)")
    ap.add_argument("--ee-halo-limit", type=float, default=2.0,
                    help="Halo suppression limiter")
    ap.add_argument("--ee-save-preview", action="store_true", default=True,
                    help="Save RGB preview after EE")

    # ---------- SAT (after EE; scale U/V only) ----------
    ap.add_argument("--sat-enable", action="store_true", default=True,
                    help="Enable saturation by scaling U/V in YUV domain")
    ap.add_argument("--sat-s", type=float, default=4.5,
                    help="Saturation scale (>1 more vivid; 0~1 desaturate)")
    ap.add_argument("--sat-save-raw", action="store_true", default=True,
                    help="Save planar .yuv after SAT")
    ap.add_argument("--sat-save-png", action="store_true", default=True,
                    help="Save Y/U/V PNG previews after SAT")
    ap.add_argument("--sat-save-rgb-preview", action="store_true", default=True,
                    help="Save YUV→RGB preview before/after SAT")

    # ---------- CONTRAST (after SAT; adjust Y') ----------
    ap.add_argument("--ctr-enable", action="store_true", default=True,
                    help="Enable Y' contrast adjustment after SAT")
    ap.add_argument("--ctr-method", type=str, default="gamma",
                    choices=["linear", "sigmoid", "gamma", "hist_eq"],
                    help="Contrast method")
    ap.add_argument("--ctr-factor", type=float, default=2.1,
                    help="linear: contrast factor (>1 increases)")
    ap.add_argument("--ctr-midpoint", type=float, default=0.5,
                    help="linear: midpoint [0,1]")
    ap.add_argument("--ctr-gain", type=float, default=8.0,
                    help="sigmoid: curve gain")
    ap.add_argument("--ctr-cutoff", type=float, default=0.5,
                    help="sigmoid: midpoint [0,1]")
    ap.add_argument("--ctr-gamma", type=float, default=1.4,
                    help="gamma: exponent")
    ap.add_argument("--ctr-save-raw", action="store_true", default=True,
                    help="Save planar .yuv after CONTRAST")
    ap.add_argument("--ctr-save-png", action="store_true", default=True,
                    help="Save Y/U/V PNG after CONTRAST")
    ap.add_argument("--ctr-save-rgb-preview", action="store_true", default=True,
                    help="Save YUV→RGB preview before/after CONTRAST")

    # ---------- HUE (after CONTRAST) ----------
    ap.add_argument("--hue-enable", action="store_true", default=True,
                    help="Enable hue adjustment after CONTRAST")
    # Global hue/sat
    ap.add_argument("--hue-deg", type=float, default=0,
                    help="Global hue rotation in degrees (CW positive)")
    ap.add_argument("--hue-sat", type=float, default=1.0,
                    help="Global saturation scale")
    # Selective hue rotation
    ap.add_argument("--hue-sel", action="store_true", default=False,
                    help="Enable selective hue rotation")
    ap.add_argument("--hue-sel-deg", type=float, default=60.0,
                    help="Selective rotation angle in degrees")
    ap.add_argument("--hue-sel-range", type=float, nargs=2, default=[180.0, 260.0],
                    help="Target hue range in degrees (e.g., 180 260 for cyan-blue)")
    ap.add_argument("--hue-sel-strength", type=float, default=0.8,
                    help="Blend strength [0,1]")
    ap.add_argument("--hue-sel-sigma", type=float, default=None,
                    help="Hue Gaussian half-width in degrees (default = half range)")
    ap.add_argument("--hue-sel-vrange", type=float, nargs=2, default=[0.4, 1.0],
                    help="Value range gate [0,1]")
    ap.add_argument("--hue-sel-smin", type=float, default=0.08,
                    help="Minimum saturation gate [0,1]")
    ap.add_argument("--hue-hiwarm", action="store_true", default=False,
                    help="Enable highlight warming")
    ap.add_argument("--hue-hi-r", type=float, default=0.04,
                    help="Highlight red relative gain")
    ap.add_argument("--hue-hi-b", type=float, default=-0.03,
                    help="Highlight blue relative gain (negative reduces blue)")
    ap.add_argument("--hue-hi-range", type=float, nargs=2, default=[0.6, 0.9],
                    help="Highlight value range [0,1]")
    ap.add_argument("--hue-save-raw", action="store_true", default=True,
                    help="Save planar .yuv after HUE")
    ap.add_argument("--hue-save-png", action="store_true", default=True,
                    help="Save Y/U/V PNG after HUE")
    ap.add_argument("--hue-save-rgb-preview", action="store_true", default=True,
                    help="Save YUV→RGB preview before/after HUE")

    # ---------- YUV2JPEG (after HUE) ----------
    ap.add_argument("--jpeg-enable", action="store_true", default=True,
                    help="Enable manual YUV→JPEG (Baseline 4:2:0)")
    ap.add_argument("--jpeg-quality", type=int, default=60,
                    help="JPEG quality (1-100)")
    ap.add_argument("--jpeg-strict-bt601", action="store_true", default=False,
                    help="If True, only accept bt601 input; else adapt")
    ap.add_argument("--jpeg-filename-stem", type=str, default="final",
                    help="Final JPEG filename stem (no extension)")
    ap.add_argument(
        "--jpeg-444-to-420",
        dest="jpeg_444_to_420",
        action="store_true",
        help="If input is 4:4:4, downsample to 4:2:0 before encoding",
    )
    ap.add_argument(
        "--no-jpeg-444-to-420",
        dest="jpeg_444_to_420",
        action="store_false",
        help="Keep 4:4:4 (no downsampling)",
    )
    ap.set_defaults(jpeg_444_to_420=False)
    return ap.parse_args()


def _expand_inputs(items):
    """Expand files/dirs/globs into a list of absolute file paths."""
    files = []
    for it in items:
        p = Path(it)
        if p.is_file():
            files.append(p.resolve())
        elif p.is_dir():
            files += [q.resolve() for q in sorted(list(p.glob("*.dng")) + list(p.glob("*.DNG")))]
        else:
            for m in glob.glob(it):
                mp = Path(m)
                if mp.is_file():
                    files.append(mp.resolve())
    return files


# ==== CCM helpers ====
def _parse_3x3_maybe(obj):
    """
    Lenient 3x3 parser: accepts [9], (3,3), (9,2) SRATIONAL, or 18 alternating (num,den).
    Returns ndarray or None.
    """
    if obj is None:
        return None
    try:
        a = np.asarray(obj, dtype=object)
        # 9 floats
        if a.ndim == 1 and a.size == 9:
            return np.asarray(a, dtype=np.float64).reshape(3, 3)
        # (3,3)
        if a.shape == (3, 3):
            return np.asarray(a, dtype=np.float64)
        # (9,2) → SRATIONAL
        if a.shape == (9, 2):
            n = np.asarray(a[:, 0], dtype=np.float64)
            d = np.asarray(a[:, 1], dtype=np.float64)
            d = np.where(d == 0, 1.0, d)
            return (n / d).reshape(3, 3)
        # 18 alternating (n,d)
        if a.ndim == 1 and a.size == 18:
            n = np.asarray(a[0::2], dtype=np.float64)
            d = np.asarray(a[1::2], dtype=np.float64)
            d = np.where(d == 0, 1.0, d)
            return (n / d).reshape(3, 3)
    except Exception:
        pass
    return None


def _inject_xyz2cam_from_meta(meta: dict, sidecar_path: Optional[Path] = None) -> dict:
    """
    Try to inject XYZ→Cam matrix to meta['ColorMatrix2'] from meta or a sidecar JSON.
    Accepts XYZ→Cam or Cam→XYZ (inverted). Records source in _ccm_matrix_source.
    """
    m = None
    src = None

    # (1) Direct XYZ→Cam
    for k in ("ColorMatrix2", "xyz2cam", "XYZ2Cam", "XYZ_to_Cam", "XYZ2CAM"):
        if k in meta and meta[k] is not None:
            m = _parse_3x3_maybe(meta[k])
            if m is not None:
                src = k
                break

    # (2) Cam→XYZ, invert if found
    if m is None:
        for k in ("cam2xyz", "Cam2XYZ", "ForwardMatrix2", "cam_to_xyz", "CAM2XYZ"):
            if k in meta and meta[k] is not None:
                m_cam2xyz = _parse_3x3_maybe(meta[k])
                if m_cam2xyz is not None:
                    try:
                        m = np.linalg.pinv(m_cam2xyz)  # pseudo-inverse for robustness
                        src = k + "^-1"
                        break
                    except Exception:
                        pass

    # (3) Sidecar JSON (<stem>.ccm.json)
    if m is None and sidecar_path and sidecar_path.exists():
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                sc = json.load(f)
            for k in ("ColorMatrix2", "xyz2cam", "XYZ2Cam", "XYZ_to_Cam"):
                if k in sc:
                    m = _parse_3x3_maybe(sc[k])
                    if m is not None:
                        src = f"sidecar:{k}"
                        break
            if m is None:
                for k in ("cam2xyz", "Cam2XYZ", "ForwardMatrix2", "cam_to_xyz"):
                    if k in sc:
                        mm = _parse_3x3_maybe(sc[k])
                        if mm is not None:
                            m = np.linalg.pinv(mm)
                            src = f"sidecar:{k}^-1"
                            break
        except Exception:
            pass

    if m is not None:
        meta2 = dict(meta)
        meta2["ColorMatrix2"] = m.astype(np.float64).reshape(3, 3).tolist()
        meta2["_ccm_matrix_source"] = src
        return meta2
    return meta


def _decide_blc_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for BLC outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_BLC") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_BLC")


def _decide_dpc_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for DPC outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_DPC") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_DPC")


def _decide_bm3d_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for BM3D outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_bm3d") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_bm3d")


def _decide_awb_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for AWB outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_AWB") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_AWB")


def _decide_demosaic_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for DEMOSAIC outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_demosaic") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_demosaic")


def _decide_ccm_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for CCM outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_CCM") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_CCM")


def _decide_gamma_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for GAMMA outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_GAMMA") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_GAMMA")


def _decide_csc_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for CSC outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_CSC") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_CSC")


# ==== New: TMC dedicated dir ====
def _decide_tmc_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for TMC outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_TMC") if centralized_outdir \
        else (in_file.parent / f"{stem}_output" / f"{stem}_TMC")

# ==== New: EE dedicated dir ====
def _decide_ee_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for EE outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_EE") if centralized_outdir \
           else (in_file.parent / f"{stem}_output" / f"{stem}_EE")
# ==== New: SAT dedicated dir ====
def _decide_sat_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for SAT outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_SAT") if centralized_outdir \
           else (in_file.parent / f"{stem}_output" / f"{stem}_SAT")
def _decide_ctr_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for CONTRAST outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_CTR") if centralized_outdir \
           else (in_file.parent / f"{stem}_output" / f"{stem}_CTR")
def _decide_hue_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for HUE outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_HUE") if centralized_outdir \
           else (in_file.parent / f"{stem}_output" / f"{stem}_HUE")

def _decide_jpeg_out_dir_for_file(in_file: Path, centralized_outdir: Optional[str]):
    """Per-file subdir for JPEG outputs."""
    stem = in_file.stem
    return (Path(centralized_outdir).resolve() / stem / f"{stem}_JPEG") if centralized_outdir \
           else (in_file.parent / f"{stem}_output" / f"{stem}_JPEG")

# ====== Stats/preview tools ======
def _hist_peak_01(x01, bins=4096):
    """Return mode-like peak location in [0,1] by histogram bin center."""
    x = np.asarray(x01, np.float32).ravel()
    if x.size == 0:
        return float("nan")
    hist, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


def _neg_fraction_per_channel(x_lin, bayer="RGGB"):
    """Compute negative-value fraction per Bayer plane."""
    bayer = (bayer or "RGGB").upper()
    offsets = {
        "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
        "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
        "GBRG": ((1, 0), (1, 1), (0, 0), (0, 1)),
        "BGGR": ((1, 1), (1, 0), (0, 1), (0, 0)),
    }[bayer]
    (yR, xR), (yGr, xGr), (yGb, xGb), (yB, xB) = offsets
    ch = {
        "R": x_lin[yR::2, xR::2],
        "Gr": x_lin[yGr::2, xGr::2],
        "Gb": x_lin[yGb::2, xGb::2],
        "B": x_lin[yB::2, xB::2],
    }
    return {k: float((v < 0).mean()) for k, v in ch.items()}


def _srgb_oetf(x01: np.ndarray) -> np.ndarray:
    """sRGB OETF from linear [0,1] to non-linear [0,1]."""
    x = np.clip(x01, 0.0, 1.0).astype(np.float32)
    a = 0.055
    thr = 0.0031308
    y = np.where(x <= thr, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)
    return np.clip(y, 0.0, 1.0)


def _bayer_preview_rgb(raw_lin: np.ndarray, bayer: str = "RGGB") -> np.ndarray:
    """Cheap Bayer mosaic preview (no demosaic), averaging the two greens."""
    bayer = (bayer or "RGGB").upper()
    H, W = raw_lin.shape[:2]
    H2, W2 = H - (H % 2), W - (W % 2)
    x = raw_lin[:H2, :W2].astype(np.float32)
    if bayer == "RGGB":
        R = x[0::2, 0::2]; Gr = x[0::2, 1::2]; Gb = x[1::2, 0::2]; B = x[1::2, 1::2]
    elif bayer == "GRBG":
        Gr = x[0::2, 0::2]; R = x[0::2, 1::2]; B = x[1::2, 0::2]; Gb = x[1::2, 1::2]
    elif bayer == "GBRG":
        Gb = x[0::2, 0::2]; B = x[0::2, 1::2]; R = x[1::2, 0::2]; Gr = x[1::2, 1::2]
    elif bayer == "BGGR":
        B = x[0::2, 0::2]; Gb = x[0::2, 1::2]; Gr = x[1::2, 0::2]; R = x[1::2, 1::2]
    else:
        R = x[0::2, 0::2]; Gr = x[0::2, 1::2]; Gb = x[1::2, 0::2]; B = x[1::2, 1::2]
    G = 0.5 * (Gr + Gb)
    rgb = np.stack([R, G, B], axis=-1)
    return rgb


def _resize_max_side_uint8(img8: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Resize uint8 image so that the longer side == max_side (no-op if smaller or max_side<=0)."""
    if max_side is None or max_side <= 0:
        return img8
    h, w = img8.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img8
    scale = max_side / float(m)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    _resampling = getattr(Image, "Resampling", Image)
    pil = Image.fromarray(img8).resize((new_w, new_h), resample=_resampling.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def save_stage_preview(x_lin: np.ndarray,
                       meta: dict,
                       out_dir: Path,
                       filename: str = "preview_srgb.png",
                       max_side: int = 1600) -> Path:
    """Quick sRGB preview saver from either Bayer (2D) or RGB (HxWx3) linear input."""
    wl = float(meta.get("white_level", 1.0))
    wl = 1.0 if wl <= 0 else wl

    if x_lin.ndim == 2:
        bayer = meta.get("bayer", "RGGB")
        rgb_lin = _bayer_preview_rgb(x_lin, bayer)
    elif x_lin.ndim == 3 and x_lin.shape[2] == 3:
        rgb_lin = x_lin.astype(np.float32)
    else:
        raise ValueError(f"save_stage_preview: 非法形状 {x_lin.shape}")

    rgb01 = np.clip(rgb_lin / wl, 0.0, 1.0)
    srgb01 = _srgb_oetf(rgb01)
    img8 = (np.clip(srgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    img8 = _resize_max_side_uint8(img8, max_side=max_side)

    out_path = Path(out_dir) / filename
    Image.fromarray(img8).save(out_path)
    return out_path


def main():
    args = parse_args()

    all_files = _expand_inputs(args.inputs)
    if not all_files:
        print("[pipeline] 没找到输入文件")
        sys.exit(1)

    centralized = args.outdir
    if centralized:
        Path(centralized).mkdir(parents=True, exist_ok=True)

    for fpath in all_files:
        fpath = Path(fpath)

        # === Load RAW + meta ===
        raw, meta = load_dng(str(fpath))
        meta = dict(meta)
        meta["file_basename"] = fpath.name
        meta["file_path"] = str(fpath)  # for CCM to read DNG tags

        # Log original size/dtype
        logger.info("[SIZE_MONITOR][LOAD] Input file: %s | Original shape: %s | dtype: %s",
                    fpath.name, raw.shape, raw.dtype)

        # === White level fallback/adjustment ===
        if "white_level" not in meta or meta["white_level"] is None:
            if np.issubdtype(raw.dtype, np.integer):
                meta["white_level"] = (
                    4095 if (raw.dtype == np.uint16 and int(raw.max()) <= 4095)
                    else int(np.iinfo(raw.dtype).max)
                )
            else:
                meta["white_level"] = 1
            logger.warning("[pipeline] meta.white_level 缺失，已兜底为 %s", meta["white_level"])

        _wl = int(meta["white_level"])
        try:
            rmax = int(raw.max())
        except Exception:
            rmax = 0

        if np.issubdtype(raw.dtype, np.floating):
            try:
                fmax = float(raw.max())
            except Exception:
                fmax = 0.0
            if fmax <= 1.5 and _wl > 4:
                logger.warning("[pipeline] 浮点 RAW(max≈%.4f) 但 WL=%d，自动调整 WL=1", fmax, _wl)
                _wl = 1
                meta["white_level"] = _wl

        BRIGHT_THRESHOLD = 0.25
        is_bright_enough = (rmax >= max(int(BRIGHT_THRESHOLD * max(_wl, 1)), 1))
        if is_bright_enough:
            if _wl > 2 * max(rmax, 1):
                logger.warning("[pipeline] meta.white_level=%d 与数据上界 max=%d 偏离较大", _wl, rmax)
        else:
            logger.info("[pipeline] 图像较暗(max=%d)，跳过 WL 偏离检查", rmax)
        WL = int(meta["white_level"])

        # === Prepare output dirs per stage ===
        blc_dir = _decide_blc_out_dir_for_file(fpath, centralized)
        dpc_dir = _decide_dpc_out_dir_for_file(fpath, centralized)
        bm3d_dir = _decide_bm3d_out_dir_for_file(fpath, centralized)
        awb_dir = _decide_awb_out_dir_for_file(fpath, centralized)
        demosaic_dir = _decide_demosaic_out_dir_for_file(fpath, centralized)
        ccm_dir = _decide_ccm_out_dir_for_file(fpath, centralized)
        gamma_dir = _decide_gamma_out_dir_for_file(fpath, centralized)
        csc_dir = _decide_csc_out_dir_for_file(fpath, centralized)
        tmc_dir = _decide_tmc_out_dir_for_file(fpath, centralized)
        ee_dir  = _decide_ee_out_dir_for_file(fpath, centralized)
        sat_dir = _decide_sat_out_dir_for_file(fpath, centralized)
        ctr_dir = _decide_ctr_out_dir_for_file(fpath, centralized)
        hue_dir = _decide_hue_out_dir_for_file(fpath, centralized)
        jpeg_dir = _decide_jpeg_out_dir_for_file(fpath, centralized)
        for d in (blc_dir, dpc_dir, bm3d_dir, awb_dir, demosaic_dir, ccm_dir, gamma_dir, csc_dir, tmc_dir, ee_dir, sat_dir, ctr_dir, hue_dir, jpeg_dir):
            d.mkdir(parents=True, exist_ok=True)

        logger.info("[pipeline] file=%s size=%dx%d bayer=%s WL=%s BL=%s",
                    fpath, raw.shape[1], raw.shape[0], meta.get('bayer'),
                    str(meta.get('white_level')), str(meta.get('black_level')))
        logger.info("[pipeline] DPC=%s | BLC=%s | bm3d=%s | AWB=%s | DEMOSAIC=%s | CCM=%s | GAMMA=%s | CSC=%s | TMC=%s | EE=%s | SAT=%s | CTR=%s | HUE=%s | jpeg=%s " ,
                    dpc_dir, blc_dir, bm3d_dir, awb_dir, demosaic_dir, ccm_dir, gamma_dir, csc_dir, tmc_dir, ee_dir, sat_dir, ctr_dir, hue_dir, jpeg_dir)

        # ---------- Step 1: DPC (updated params pass-through) ----------
        logger.info("[pipeline] ===> 进入 DPC")
        try:
            dpc_modes = [m.strip() for m in str(args.dpc_modes).split(",") if m.strip()]

            # Pass CLI args to DPC; it decides thresholds/noise internally
            blc_input_raw, bad_mask = run_dpc_on_single_raw(
                raw, meta, str(dpc_dir),
                modes=dpc_modes,
                thres=args.dpc_thres,                 # float for abs/rel
                thres_mode=str(args.dpc_thres_mode),  # 'abs' / 'rel' / 'auto'
                auto_k=float(args.dpc_auto_k),
                bad_neighbor_min=int(args.dpc_bad_min),
                num_bad_pixels=int(args.dpc_bad_num),
                pick_for_blc=str(args.dpc_pick)
            )
            if not isinstance(blc_input_raw, np.ndarray):
                raise TypeError("run_dpc_on_single_raw() 必须返回 ndarray 作为 BLC 输入")

            monitor_size("DPC", raw, blc_input_raw, meta)

        except Exception as e:
            logger.error("[pipeline][DPC] 运行失败：%s", e)
            logger.warning("[pipeline][DPC] 回退使用原始 RAW 进入 BLC，坏点掩码置空。")
            blc_input_raw, bad_mask = raw, None
        logger.info("[pipeline] <=== 完成 DPC")

        try:
            save_stage_preview(blc_input_raw, meta, dpc_dir, filename="preview_srgb.png")
        except Exception as e:
            logger.warning("[pipeline][DPC] 预览保存失败：%s", e)

        # ---------- Step 2: BLC ----------
        logger.info("[pipeline] ===> 进入 BLC")
        raw_blc_lin, meta_blc, blc_info = run_auto_blc(
            blc_input_raw, meta, str(blc_dir),
            p_list=args.p_list, scale_list=args.scale_list,
            neg_max=args.neg_max,
            save_plots=True,
            return_mode="lin",
            exclude_mask=bad_mask
        )
        monitor_size("BLC", blc_input_raw, raw_blc_lin, meta_blc)
        logger.info("[pipeline] <=== 完成 BLC")

        try:
            save_stage_preview(raw_blc_lin, meta_blc, blc_dir, filename="preview_srgb.png")
        except Exception as e:
            logger.warning("[pipeline][BLC] 预览保存失败：%s", e)

        WL = float(meta_blc.get("white_level", 4095.0))
        inp01 = np.clip(blc_input_raw.astype(np.float32) / max(WL, 1e-6), 0, 1)
        out01 = np.clip(raw_blc_lin.astype(np.float32) / max(WL, 1e-6), 0, 1)
        hp_in, hp_out = _hist_peak_01(inp01), _hist_peak_01(out01)
        q001_in, q005_in = float(np.quantile(inp01, 0.001)), float(np.quantile(inp01, 0.005))
        q001_out, q005_out = float(np.quantile(out01, 0.001)), float(np.quantile(out01, 0.005))
        neg_all = float((raw_blc_lin < 0).mean())
        neg_ch = _neg_fraction_per_channel(raw_blc_lin, meta_blc.get("bayer", "RGGB"))
        bl_used = {k: float(v) for k, v in blc_info["bl_used"].items()}
        best = blc_info["best_record"]
        logger.info("[BLC/summary] file=%s | WL=%.1f | origin=%s p=%s scale=%.3f | BL_used=%s",
                    meta_blc.get("file_basename"), WL, best.get("origin"),
                    ("%.3f" % best["estimate_p"]) if best.get("estimate_p") is not None else "meta",
                    best["scale"], json.dumps(bl_used))
        logger.info("[BLC/metrics] peak(in→out)=%.5f→%.5f | q001=%.6f→%.6f | q005=%.6f→%.6f",
                    hp_in, hp_out, q001_in, q001_out, q005_in, q005_out)
        logger.info("[BLC/neg] neg_all=%.6f | per_ch: R=%.6f Gr=%.6f Gb=%.6f B=%.6f | neg_max=%.6f",
                    neg_all, neg_ch["R"], neg_ch["Gr"], neg_ch["Gb"], neg_ch["B"], float(args.neg_max))

        # ---------- Step 3: BM3D ----------
        logger.info("[pipeline] ===> 进入 BM3D")
        # Skip flag: python pipeline.py <inputs> --skip-bm3d
        if bool(args.skip_bm3d):
            logger.info("[pipeline][BM3D] 已按参数 --skip-bm3d 跳过 BM3D；BLC 输出将直接进入 AWB。")
            raw_dn_lin, meta_dn, dn_info = raw_blc_lin, dict(meta_blc), {"skipped": True, "fallback": "blc"}
            try:
                monitor_size("BM3D(SKIPPED)", raw_blc_lin, raw_dn_lin, meta_dn)
            except Exception:
                pass
        else:
            try:
                raw_dn_lin, meta_dn, dn_info = run_bm3d_raw(
                    raw_blc_lin, meta_blc, str(bm3d_dir),
                    method=args.bm3d_method,
                    strength=float(args.bm3d_strength),
                    sigma=None,
                    save_plots=bool(args.bm3d_save_plots),
                    save_exact16=False
                )
                monitor_size("BM3D", raw_blc_lin, raw_dn_lin, meta_dn)
            except Exception as e:
                logger.error("[pipeline][BM3D] 运行失败：%s", e)
                logger.error("[pipeline][BM3D] 堆栈：%s", traceback.format_exc())
                raw_dn_lin, meta_dn, dn_info = raw_blc_lin, dict(meta_blc), {"fallback": "blc"}
        logger.info("[pipeline] <=== 完成 BM3D")

        try:
            save_stage_preview(raw_dn_lin, meta_dn, bm3d_dir, filename="preview_srgb.png")
        except Exception as e:
            logger.warning("[pipeline][BM3D] 预览保存失败：%s", e)

        # ---------- Step 4: AWB ----------
        logger.info("[pipeline] ===> 进入 AWB")
        try:
            raw_awb_lin, meta_awb, awb_info = run_awb_raw(
                raw_dn_lin, meta_dn, str(awb_dir),
                top_percent=args.awb_top_percent,
                lower_frac=float(args.awb_lower_frac),
                upper_frac=float(args.awb_upper_frac),
                save_plots=bool(args.awb_save_plots)
            )
            monitor_size("AWB", raw_dn_lin, raw_awb_lin, meta_awb)
        except Exception as e:
            logger.error("[pipeline][AWB] 运行失败：%s", e)
            logger.error("[pipeline][AWB] 堆栈：%s", traceback.format_exc())
            raw_awb_lin, meta_awb, awb_info = raw_dn_lin, dict(meta_dn), {"fallback": "bm3d"}

        try:
            save_stage_preview(raw_awb_lin, meta_awb, awb_dir, filename="preview_srgb.png")
        except Exception as e:
            logger.warning("[pipeline][AWB] 预览保存失败：%s", e)

        g = awb_info.get("gains", [np.nan, np.nan, np.nan, np.nan]) if isinstance(awb_info, dict) else [np.nan] * 4
        logger.info("[AWB/summary] top=%.3f%% | gains R=%.4f G=%.4f B=%.4f",
                    float(awb_info.get("eff_top_percent", float("nan"))) if isinstance(awb_info, dict) else float("nan"),
                    g[0], g[1], g[2])
        pre = awb_info.get("pre_means_all", {}) if isinstance(awb_info, dict) else {}
        post = awb_info.get("post_means_all", {}) if isinstance(awb_info, dict) else {}
        logger.info("[AWB/means] pre(R,G,B)=(%.2f,%.2f,%.2f) → post(R,G,B)=(%.2f,%.2f,%.2f)",
                    float(pre.get("R", np.nan)), float(pre.get("G_avg", np.nan)), float(pre.get("B", np.nan)),
                    float(post.get("R", np.nan)), float(post.get("G_avg", np.nan)), float(post.get("B", np.nan)))

        # ---------- Step 5: DEMOSAIC ----------
        raw_awb = raw_awb_lin.astype(np.float32)
        m = _bayer_masks(*raw_awb.shape, meta_awb.get("bayer", "RGGB"))
        Rm = float(raw_awb[m.R  > 0].mean())
        Grm = float(raw_awb[m.Gr > 0].mean())
        Gbm = float(raw_awb[m.Gb > 0].mean())
        Bm = float(raw_awb[m.B  > 0].mean())
        Gm = 0.5 * (Grm + Gbm)
        logger.info("AWB mosaic means: R=%.3f  G=%.3f  B=%.3f  |  R/G=%.3f  B/G=%.3f",
                    Rm, Gm, Bm, (Rm/(Gm+1e-6)), (Bm/(Gm+1e-6)))

        # Pass BM3D sigma (scaled by AWB gains) for more stable demosaic adaptation
        sigma_base = dn_info.get("sigma_after") or dn_info.get("sigma_used") if isinstance(dn_info, dict) else None
        sigma_rgba = None
        if isinstance(sigma_base, dict):
            gR, gG, gB = map(float, awb_info.get("gains", [1.0, 1.0, 1.0])[:3]) if isinstance(awb_info, dict) else (1.0,1.0,1.0)
            sigma_rgba = (float(sigma_base.get("R",0.0))*gR,
                          float(sigma_base.get("Gr",0.0))*gG,
                          float(sigma_base.get("Gb",0.0))*gG,
                          float(sigma_base.get("B",0.0))*gB)
            logger.info("[BM3D→DEMOSAIC] sigma_tuple(R,Gr,Gb,B)=%s", tuple(round(v,3) for v in sigma_rgba))
        else:
            logger.info("[BM3D→DEMOSAIC] 未获取到 sigma，demosaic 使用内部估计。")

        logger.info("[pipeline] ===> 进入 DEMOSAIC (AUTO)")
        try:
            # "Brighter & color-preserving" configuration
            rgb_lin, meta_rgb, info = run_demosaic_raw(
                raw_awb_lin, meta_awb, str(demosaic_dir),
                bayer=meta.get("bayer", "RGGB"),
                mhc_only=False,
                k_sigma=4.0,

                # Preserve chroma; light smoothing
                moire_chroma_atten=1.0,
                gf_radius=1,
                ratio_min=0.05, ratio_max=16.0,

                # Light USM on luminance
                usm_amount=0.12, usm_radius=1, usm_thresh=0.0,

                save_plots=True
            )

            # AWB color/brightness lock back to demosaic result
            R2 = float(rgb_lin[...,0].mean())
            G2 = float(rgb_lin[...,1].mean())
            B2 = float(rgb_lin[...,2].mean())

            sG = (Gm + 1e-6) / (G2 + 1e-6)                  # brightness lock
            kR = ((Rm/(Gm+1e-6)) / (R2/(G2+1e-6) + 1e-12))  # R/G lock
            kB = ((Bm/(Gm+1e-6)) / (B2/(G2+1e-6) + 1e-12))  # B/G lock
            gain_diag = np.array([sG*kR, sG*1.0, sG*kB], dtype=np.float32).reshape(1,1,3)
            rgb_lin = rgb_lin * gain_diag

            WL_dem = float(meta_awb.get("white_level", 4095.0))
            rgb_lin = np.clip(rgb_lin, 0.0, WL_dem)

            meta_rgb = dict(meta_rgb)
            meta_rgb["white_level"] = meta_awb.get("white_level", meta_rgb.get("white_level"))
            meta_rgb["black_level"] = meta_awb.get("black_level", meta_rgb.get("black_level"))
            meta_rgb["_color_lock_diag"] = [float(g) for g in gain_diag.ravel().tolist()]

            monitor_size("DEMOSAIC", raw_awb_lin, rgb_lin, meta_rgb)
            logger.info("[COLOR-LOCK] sG=%.6f, kR=%.6f, kB=%.6f | "
                        "mosaic R/G=%.4f B/G=%.4f | demosaiced(pre) R/G=%.4f B/G=%.4f",
                        sG, kR, kB,
                        (Rm/(Gm+1e-6)), (Bm/(Gm+1e-6)),
                        (R2/(G2+1e-6)), (B2/(G2+1e-6)))

        except Exception as e:
            logger.error("[pipeline][DEMOSAIC] 运行失败：%s", e)
            logger.error("[pipeline][DEMOSAIC] 堆栈：%s", traceback.format_exc())
            rgb_lin, meta_rgb, info = None, None, None
        logger.info("[pipeline] <=== 完成 DEMOSAIC")

        if rgb_lin is not None and meta_rgb is not None:
            try:
                save_stage_preview(rgb_lin, meta_rgb, demosaic_dir, filename="preview_srgb.png")
            except Exception as e:
                logger.warning("[pipeline][DEMOSAIC] 预览保存失败：%s", e)

        # ---------- Step 6: CCM (after DEMOSAIC) ----------
        if rgb_lin is not None and meta_rgb is not None:
            logger.info("[pipeline] ===> 进入 CCM (Cam->%s, %s)",
                        str(args.ccm_color_space).upper(), str(args.ccm_illuminant).upper())

            # Optional look matrix/bias
            L = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]
            b = [0.001, 0.0001, 0.0001]

            # Try sidecar/meta-based XYZ→Cam injection
            sidecar = fpath.with_suffix(".ccm.json")
            meta_rgb = _inject_xyz2cam_from_meta(meta_rgb, sidecar if sidecar.exists() else None)
            if "_ccm_matrix_source" in meta_rgb:
                logger.info("[CCM] 注入矩阵来源: %s", meta_rgb["_ccm_matrix_source"])

            try:
                rgb_before_ccm = rgb_lin

                rgb_ccm_lin, meta_ccm, ccm_info = run_ccm_rgb(
                    rgb_lin, meta_rgb, str(ccm_dir),
                    color_space=str(args.ccm_color_space),
                    illuminant=str(args.ccm_illuminant),
                    save_json=True,
                    fallback_strategy="assume_xyz_d65",
                    fallback_blend=0.10,
                    # post_look_matrix = L,
                    # post_look_bias = b,
                    # post_look_strength = 0.6,
                    # post_look_preserve_neutral = False
                )

                monitor_size("CCM", rgb_before_ccm, rgb_ccm_lin, meta_ccm)

                logger.info("[CCM] color_space=%s illuminant=%s used_tags=%s",
                            ccm_info.get("color_space"), ccm_info.get("illuminant"),
                            ccm_info.get("used_tags"))

                rgb_lin, meta_rgb = rgb_ccm_lin, meta_ccm

            except Exception as e:
                logger.error("[pipeline][CCM] 运行失败：%s", e)
                logger.error("[pipeline][CCM] 堆栈：%s", traceback.format_exc())
                rgb_ccm_lin, meta_ccm, ccm_info = rgb_lin, dict(meta_rgb), {"fallback": "identity"}

            try:
                save_stage_preview(rgb_ccm_lin, meta_ccm, ccm_dir, filename="preview_srgb.png")
            except Exception as e:
                logger.warning("[pipeline][CCM] 预览保存失败：%s", e)

        # ---------- Step 7: GAMMA ----------
        if rgb_lin is not None and meta_rgb is not None:
            logger.info("[pipeline] ===> 进入 GAMMA")
            try:
                rgb_before_gamma = rgb_lin

                rgb_gamma, meta_gamma, gamma_info = run_gamma_rgb(
                    rgb_in=rgb_lin,
                    meta_in=meta_rgb,
                    out_dir=str(gamma_dir),
                    gamma=float(args.gamma),
                    save_bitdepth=int(args.gamma_save_bitdepth) if args.gamma_save_bitdepth else None,
                    clamp=not bool(args.gamma_no_clamp),
                    save_filename="rgb_gamma",
                )

                monitor_size("GAMMA", rgb_before_gamma, rgb_gamma, meta_gamma)

                logger.info("[GAMMA/summary] gamma=%.4f | clamp=%s | WL(in→out)=%.1f→%.1f",
                            float(gamma_info["gamma"]), str(gamma_info["clamp"]),
                            float(gamma_info["white_level_in"]), float(gamma_info["white_level_out"]))
                if gamma_info.get("saved"):
                    logger.info("[GAMMA] 输出已保存：%s（%s-bit）",
                                gamma_info.get("saved_path"), int(gamma_info.get("saved_bitdepth", 0)))
                if bool(args.gamma_save_plots):
                    try:
                        from_this = save_stage_preview(rgb_lin, meta_rgb, gamma_dir, filename="preview_before_srgb.png")
                        to_this = save_stage_preview(rgb_gamma, meta_gamma, gamma_dir,
                                                     filename="preview_after_srgb.png")
                        logger.info("[GAMMA] 预览保存：%s | %s", from_this, to_this)
                    except Exception as e:
                        logger.warning("[pipeline][GAMMA] 预览保存失败：%s", e)
                rgb_lin, meta_rgb = rgb_gamma, meta_gamma
            except Exception as e:
                logger.error("[pipeline][GAMMA] 运行失败：%s", e)
                logger.error("[pipeline][GAMMA] 堆栈：%s", traceback.format_exc())
            logger.info("[pipeline] <=== 完成 GAMMA")

        # — Ensure CSC input is R'G'B' uint8 —
        def _ensure_rgb_prime_u8(x, meta):
            """Normalize to R'G'B' uint8 using sRGB OETF."""
            wl = float(meta.get("white_level", 1.0)) or 1.0
            x01 = np.clip(x.astype(np.float32) / wl, 0.0, 1.0)

            a = 0.055
            thr = 0.0031308
            y01 = np.where(x01 <= thr, 12.92 * x01, (1 + a) * np.power(x01, 1 / 2.4) - a)

            return (np.clip(y01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        # ---------- Step 8: CSC (RGB → YCbCr444 8-bit) ----------
        csc_ok = False
        Y = U = V = None

        if bool(args.csc_enable) and (rgb_lin is not None and meta_rgb is not None):
            logger.info("[pipeline] ===> 进入 CSC (RGB' -> YCbCr444, %s)", str(args.csc_standard).upper())
            try:
                rgb_before_csc = rgb_lin

                # Ensure CSC input is R'G'B' uint8
                rgb_for_csc_u8 = _ensure_rgb_prime_u8(rgb_lin, meta_rgb)

                (Y, U, V), meta_csc, csc_info = run_csc_rgb_to_yuv444(
                    rgb_in=rgb_for_csc_u8,
                    meta_in=meta_rgb,
                    out_dir=str(csc_dir),
                    standard=str(args.csc_standard),
                    range_mode="full",
                    clamp=True,
                    save_raw=bool(args.csc_save_raw),
                    save_png=bool(args.csc_save_png),
                    filename_stem=str(args.csc_filename_stem),
                )

                # Monitor tuple output
                monitor_size("CSC", rgb_before_csc, (Y, U, V), meta_csc)

                # Propagate CSC info for previews
                try:
                    if isinstance(csc_info, dict) and "coeffs" in csc_info:
                        meta_csc["coeffs"] = csc_info["coeffs"]
                    if isinstance(csc_info, dict) and "standard" in csc_info:
                        meta_csc["csc_standard"] = csc_info["standard"]
                    if isinstance(csc_info, dict) and "range" in csc_info:
                        meta_csc["csc_range"] = csc_info["range"]
                except Exception:
                    pass
                meta_csc["range"] = meta_csc.get("csc_range", "full")
                meta_csc["uv_center"] = 0.5

                logger.info("[CSC/summary] 保存路径：%s", json.dumps(csc_info.get("saved", {}), ensure_ascii=False))

                # Optional YUV→RGB preview
                if bool(args.csc_save_rgb_preview):
                    try:
                        csc_standard = meta_csc.get("csc_standard", str(args.csc_standard))
                        csc_range = meta_csc.get("csc_range", "full")
                        preview_path = save_yuv_as_rgb_preview(
                            Y, U, V,
                            meta_csc,
                            csc_dir,
                            filename="preview_yuv2rgb.png",
                            standard=csc_standard,
                            range_mode=csc_range
                        )
                        logger.info("[CSC] YUV转RGB预览已保存：%s", preview_path)
                    except Exception as e:
                        logger.warning("[pipeline][CSC] YUV转RGB预览保存失败：%s", e)

                csc_ok = True
            except Exception as e:
                logger.error("[pipeline][CSC] 运行失败：%s", e)
                logger.error("[pipeline][CSC] 堆栈：%s", traceback.format_exc())
            logger.info("[pipeline] <=== 完成 CSC")
            # Debug ranges
            try:
                logger.info(f"[YUV2RGB Debug] Y range: [{float(Y.min()):.1f}, {float(Y.max()):.1f}]")
                logger.info(f"[YUV2RGB Debug] U range: [{float(U.min()):.1f}, {float(U.max()):.1f}]")
                logger.info(f"[YUV2RGB Debug] V range: [{float(V.min()):.1f}, {float(V.max()):.1f}]")
                logger.info(f"[YUV2RGB Debug] U mean: {float(U.mean()):.1f}, V mean: {float(V.mean()):.1f}")
            except Exception:
                pass

        # ---------- Step 9: TMC (after CSC; Y' only) ----------
        tmc_ok = False
        if csc_ok and bool(args.tmc_enable) and Y is not None and U is not None and V is not None:
            logger.info("[pipeline] ===> 进入 TMC (after CSC; tone-map Y' only; out=%s)", str(tmc_dir))
            try:
                curve_arg = None if str(args.tmc_curve).lower() == "auto" else str(args.tmc_curve).lower()

                yuv_before_tmc = (Y, U, V)

                (Yt, Ut, Vt), meta_tmc, tmc_info = run_tmc_after_csc(
                    Y_in=Y, U_in=U, V_in=V,
                    meta_in=meta_csc,
                    out_dir=str(tmc_dir),
                    range_mode=meta_csc.get("csc_range", "full"),
                    curve=curve_arg,
                    enable_local_lite=True,
                    local_radius=int(args.tmc_local_radius),
                    local_detail_pow=float(args.tmc_local_detail_pow),
                    save_preview_rgb=bool(args.tmc_save_preview),
                    filename_stem=str(args.csc_filename_stem)
                )
                tmc_ok = True
                monitor_size("TMC", yuv_before_tmc, (Yt, Ut, Vt), meta_tmc)

                for k in ("coeffs", "csc_standard", "csc_range", "range", "uv_center"):
                    if k not in meta_tmc and k in meta_csc:
                        meta_tmc[k] = meta_csc[k]

                logger.info("[TMC/summary] 保存路径：%s", json.dumps(tmc_info.get("saved", {}), ensure_ascii=False))

                if bool(args.tmc_save_rgb_preview):
                    try:
                        tmc_standard = meta_tmc.get("csc_standard", str(args.csc_standard))
                        tmc_range = meta_tmc.get("csc_range", "full")
                        preview_path = save_yuv_as_rgb_preview(
                            Yt, Ut, Vt,
                            meta_tmc,
                            tmc_dir,
                            filename="preview_tmc_yuv2rgb.png",
                            standard=tmc_standard,
                            range_mode=tmc_range
                        )
                        logger.info("[TMC] YUV转RGB预览已保存：%s", preview_path)

                        preview_before_path = save_yuv_as_rgb_preview(
                            Y, U, V,
                            meta_tmc,
                            tmc_dir,
                            filename="preview_before_tmc_yuv2rgb.png",
                            standard=tmc_standard,
                            range_mode=tmc_range
                        )
                        logger.info("[TMC] TMC前YUV转RGB预览已保存：%s", preview_before_path)
                    except Exception as e:
                        logger.warning("[pipeline][TMC] YUV转RGB预览保存失败：%s", e)

            except Exception as e:
                logger.error("[pipeline][TMC] 运行失败：%s", e)
                logger.error("[pipeline][TMC] 堆栈：%s", traceback.format_exc())
            logger.info("[pipeline] <=== 完成 TMC")

        # ---------- Step 10: EE (after TMC; Y' sharpen) ----------
        ee_ok = False
        if csc_ok and tmc_ok and bool(args.ee_enable):
            logger.info("[pipeline] ===> 进入 EE (after TMC; sharpen Y’ only; out=%s)", str(ee_dir))
            try:
                ee_standard = meta_tmc.get("csc_standard", str(args.csc_standard))
                ee_range    = meta_tmc.get("csc_range", "full")

                try:
                    save_yuv_as_rgb_preview(
                        Yt, Ut, Vt, meta_tmc, ee_dir,
                        filename="preview_before_ee_yuv2rgb.png",
                        standard=ee_standard, range_mode=ee_range,
                        max_side=0
                    )
                    ee_ok = True
                except Exception as e:
                    logger.warning("[EE] 保存 EE 前预览失败：%s", e)

                ee_stem = f"{str(args.csc_filename_stem)}_ee"

                (Ys, Us, Vs), meta_ee, ee_info = run_edge_enhance_after_tmc(
                    Y_in=Yt, U_in=Ut, V_in=Vt,
                    meta_in=meta_tmc,
                    out_dir=str(ee_dir),
                    amount=float(args.ee_amount),
                    radius=int(args.ee_radius),
                    threshold=float(args.ee_threshold),
                    halo_limit=float(args.ee_halo_limit),
                    save_preview_rgb=bool(args.ee_save_preview),
                    filename_stem=ee_stem
                )

                # Quick delta stats for sanity
                try:
                    dY = Ys.astype(np.float32) - Yt.astype(np.float32)
                    abs_mean = float(np.mean(np.abs(dY)))
                    abs_max  = float(np.max(np.abs(dY)))
                    changed_ratio = float((np.abs(dY) >= 1.0).mean())
                    logger.info("[EE/delta] |Y_out - Y_in| mean=%.6f, max=%.6f, changed(>=1)=%0.3f%%",
                                abs_mean, abs_max, changed_ratio * 100.0)
                except Exception:
                    pass

                try:
                    ee_after_path = save_yuv_as_rgb_preview(
                        Ys, Us, Vs, meta_ee, ee_dir,
                        filename="preview_after_ee_yuv2rgb.png",
                        standard=ee_standard, range_mode=ee_range,
                        max_side=0
                    )
                    logger.info("[EE] 已保存 EE 后预览：%s", ee_after_path)
                except Exception as e:
                    logger.warning("[EE] 保存 EE 后预览失败：%s", e)

                monitor_size("EE", (Yt, Ut, Vt), (Ys, Us, Vs), meta_ee)

                logger.info("[EE/summary] sigma=%.5f thr_eff=%.5f amount=%.3f radius=%d halo=%.2f",
                            float(ee_info.get("sigma_est", float("nan"))),
                            float(ee_info.get("thr_eff", float("nan"))),
                            float(args.ee_amount), int(args.ee_radius), float(args.ee_halo_limit))
            except Exception as e:
                logger.error("[pipeline][EE] 运行失败：%s", e)
                logger.error("[pipeline][EE] 堆栈：%s", traceback.format_exc())
            logger.info("[pipeline] <=== 完成 EE")

        # ---------- Step 11: SAT (after EE; scale U/V) ----------
        sat_ok = False
        if csc_ok and tmc_ok and ee_ok and bool(args.sat_enable):
            logger.info("[pipeline] ===> 进入 SAT (after EE; scale U/V by s; out=%s)", str(sat_dir))
            try:
                sat_standard = meta_ee.get("csc_standard", str(args.csc_standard))
                sat_range    = meta_ee.get("csc_range", "full")
                sat_stem     = f"{str(args.csc_filename_stem)}_sat"

                (Yk, Uk, Vk), meta_sat, sat_info = run_sat_after_ee(
                    Y_in=Ys, U_in=Us, V_in=Vs,
                    meta_in=meta_ee,
                    out_dir=str(sat_dir),
                    s=float(args.sat_s),
                    range_mode=sat_range,
                    standard=sat_standard,
                    save_raw=bool(args.sat_save_raw),
                    save_png=bool(args.sat_save_png),
                    save_preview_rgb=bool(args.sat_save_rgb_preview),
                    filename_stem=sat_stem
                )
                sat_ok = True

                monitor_size("SAT", (Ys, Us, Vs), (Yk, Uk, Vk), meta_sat)

                logger.info("[SAT/summary] s=%.3f | standard=%s | range=%s | saved=%s",
                            float(args.sat_s), sat_standard, sat_range,
                            {k: ('...' if isinstance(v, dict) else v) for k, v in sat_info.get("saved", {}).items()})
            except Exception as e:
                logger.error("[pipeline][SAT] 运行失败：%s", e)
                logger.error("[pipeline][SAT] 堆栈：%s", traceback.format_exc())
            logger.info("[pipeline] <=== 完成 SAT")

        # ---------- Step 12: CONTRAST (after SAT; Y' only) ----------
        ctr_ok = False
        if csc_ok and tmc_ok and ee_ok and bool(args.ctr_enable):
            logger.info("[pipeline] ===> 进入 CONTRAST (after SAT; adjust Y’ only; out=%s)", str(ctr_dir))
            try:
                # Prefer SAT output if available; otherwise EE
                if 'sat_ok' in locals() and sat_ok:
                    Yin, Uin, Vin = Yk, Uk, Vk
                    meta_in = meta_sat if 'meta_sat' in locals() else meta_ee
                    ctr_stem = f"{str(args.csc_filename_stem)}_sat_ctr"
                else:
                    Yin, Uin, Vin = Ys, Us, Vs
                    meta_in = meta_ee
                    ctr_stem = f"{str(args.csc_filename_stem)}_ctr"

                ctr_standard = meta_in.get("csc_standard", str(args.csc_standard))
                ctr_range    = meta_in.get("csc_range", "full")

                (Yc, Uc, Vc), meta_ctr, ctr_info = run_contrast_after_sat(
                    Y_in=Yin, U_in=Uin, V_in=Vin,
                    meta_in=meta_in,
                    out_dir=str(ctr_dir),
                    method=str(args.ctr_method),
                    factor=float(args.ctr_factor),
                    midpoint=float(args.ctr_midpoint),
                    gain=float(args.ctr_gain),
                    cutoff=float(args.ctr_cutoff),
                    gamma=float(args.ctr_gamma),
                    range_mode=ctr_range,
                    standard=ctr_standard,
                    save_raw=bool(args.ctr_save_raw),
                    save_png=bool(args.ctr_save_png),
                    save_preview_rgb=bool(args.ctr_save_rgb_preview),
                    filename_stem=ctr_stem
                )
                ctr_ok = True

                # Extra full-res preview
                try:
                    ctr_standard = meta_ctr.get("csc_standard", str(args.csc_standard))
                    ctr_range = meta_ctr.get("csc_range", "full")
                    save_yuv_as_rgb_preview(
                        Yc, Uc, Vc, meta_ctr, ctr_dir,
                        filename="preview_after_ctr_yuv2rgb_full.png",
                        standard=ctr_standard, range_mode=ctr_range,
                        max_side=0
                    )
                    logger.info("[CONTRAST] 已保存全分辨率预览: preview_after_ctr_yuv2rgb_full.png")
                except Exception as e:
                    logger.warning("[CONTRAST] 保存全分辨率预览失败：%s", e)

                monitor_size("CONTRAST", (Yin, Uin, Vin), (Yc, Uc, Vc), meta_ctr)

                logger.info("[CONTRAST/summary] method=%s | params=%s | range=%s | saved=%s",
                            str(args.ctr_method),
                            {k: ctr_info['params'][k] for k in ctr_info.get('params', {})},
                            ctr_range,
                            {k: ('...' if isinstance(v, dict) else v) for k, v in ctr_info.get("saved", {}).items()})
            except Exception as e:
                logger.error("[pipeline][CONTRAST] 运行失败：%s", e)
                logger.error("[pipeline][CONTRAST] 堆栈：%s", traceback.format_exc())
            logger.info("[pipeline] <=== 完成 CONTRAST")

        # ---------- Step 13: HUE (after CONTRAST) ----------
        hue_ok = False
        if csc_ok and tmc_ok and ee_ok and bool(args.hue_enable):
            logger.info("[pipeline] ===> 进入 HUE (after CONTRAST; out=%s)", str(hue_dir))
            try:
                # Choose input source: CTR > SAT > EE
                if 'ctr_ok' in locals() and ctr_ok:
                    Yin, Uin, Vin = Yc, Uc, Vc
                    meta_in = meta_ctr if 'meta_ctr' in locals() else meta_sat if 'meta_sat' in locals() else meta_ee
                    hue_stem = f"{str(args.csc_filename_stem)}_ctr_hue"
                elif 'sat_ok' in locals() and sat_ok:
                    Yin, Uin, Vin = Yk, Uk, Vk
                    meta_in = meta_sat if 'meta_sat' in locals() else meta_ee
                    hue_stem = f"{str(args.csc_filename_stem)}__sat_hue"
                else:
                    Yin, Uin, Vin = Ys, Us, Vs
                    meta_in = meta_ee
                    hue_stem = f"{str(args.csc_filename_stem)}_hue"

                hue_standard = meta_in.get("csc_standard", str(args.csc_standard))
                hue_range = meta_in.get("csc_range", "full")

                (Yh, Uh, Vh), meta_hue, hue_info = run_hue_after_contrast(
                    Y_in=Yin, U_in=Uin, V_in=Vin,
                    meta_in=meta_in,
                    out_dir=str(hue_dir),
                    # global
                    deg_global=float(args.hue_deg),
                    sat_global=float(args.hue_sat),
                    # selective
                    enable_selective=bool(args.hue_sel),
                    sel_deg=float(args.hue_sel_deg),
                    sel_hue_range=tuple(args.hue_sel_range),
                    sel_strength=float(args.hue_sel_strength),
                    sel_sigma=float(args.hue_sel_sigma) if args.hue_sel_sigma else None,
                    sel_v_range=tuple(args.hue_sel_vrange),
                    sel_s_min=float(args.hue_sel_smin),
                    # highlight warm
                    enable_hi_warm=bool(args.hue_hiwarm),
                    hi_r_shift=float(args.hue_hi_r),
                    hi_b_shift=float(args.hue_hi_b),
                    hi_v_range=tuple(args.hue_hi_range),
                    # save options
                    range_mode=hue_range,
                    standard=hue_standard,
                    save_raw=bool(args.hue_save_raw),
                    save_png=bool(args.hue_save_png),
                    save_preview_rgb=bool(args.hue_save_rgb_preview),
                    filename_stem=hue_stem
                )
                hue_ok = True

                monitor_size("HUE", (Yin, Uin, Vin), (Yh, Uh, Vh), meta_hue)

                logger.info("[HUE/summary] deg=%.2f sat=%.3f | selective=%s | hi_warm=%s | saved=%s",
                            float(args.hue_deg), float(args.hue_sat),
                            bool(args.hue_sel), bool(args.hue_hiwarm),
                            {k: ('...' if isinstance(v, dict) else v)
                             for k, v in hue_info.get("saved", {}).items()})

                if args.hue_sel and "selective" in hue_info:
                    logger.info("[HUE/selective] deg=%.2f range=[%.1f,%.1f] strength=%.2f",
                                hue_info["selective"]["deg"],
                                hue_info["selective"]["hue_range"][0],
                                hue_info["selective"]["hue_range"][1],
                                hue_info["selective"].get("strength", 0.0))

            except Exception as e:
                logger.error("[pipeline][HUE] 运行失败：%s", e)
                logger.error("[pipeline][HUE] 堆栈：%s", traceback.format_exc())
            logger.info("[pipeline] <=== 完成 HUE")

            # ---------- FINAL: manual YUV → JPEG ----------
            if bool(args.jpeg_enable) and csc_ok and tmc_ok and ee_ok and hue_ok:
                logger.info("[pipeline] ===> 进入 YUV2JPEG (manual encoder; out=%s)", str(jpeg_dir))
                try:
                    Yin, Uin, Vin = Yh, Uh, Vh
                    meta_in = meta_hue

                    info = run_yuv2jpeg_manual(
                        Y_in=Yin, U_in=Uin, V_in=Vin, meta_in=meta_in,
                        out_dir=str(jpeg_dir),
                        filename_stem=str(args.jpeg_filename_stem),
                        quality=int(args.jpeg_quality) if hasattr(args, "jpeg_quality") else 50,
                        adapt_non_bt601=False,
                        downsample_444_to_420=bool(getattr(args, "jpeg_444_to_420", False)),

                        # 4:4:4 options: use luma QTable for chroma
                        use_luma_q_for_chroma_when_444=bool(getattr(args, "jpeg_chroma_444_use_luma_q", True)),
                        chroma_q_scale_when_444=float(getattr(args, "jpeg_chroma_444_q_scale", 1.0)),
                    )

                    rep = info.get("report", {})
                    logger.info(
                        "[YUV2JPEG] saved=%s | std_in=%s | range_in=%s | ops=%s | q=%d | ss_in=%s | ss_out=%s",
                        info["saved_path"],
                        rep.get("in_standard"),
                        rep.get("in_range"),
                        ",".join(rep.get("ops", [])),
                        rep.get("quality", -1),
                        rep.get("in_subsampling", ""),
                        rep.get("out_subsampling", ""),
                    )
                except Exception as e:
                    logger.error("[pipeline][YUV2JPEG] 运行失败：%s", e)
                    logger.error("[pipeline][YUV2JPEG] 堆栈：%s", traceback.format_exc())
                logger.info("[pipeline] <=== 完成 YUV2JPEG")

        # Final summary banner
        logger.info("=" * 70)
        logger.info("[SIZE_MONITOR][FINAL] Pipeline completed for %s", fpath.name)
        logger.info("=" * 70)

    logger.info("[pipeline] All done.")


if __name__ == "__main__":
    main()
