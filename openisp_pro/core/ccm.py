# core/ccm.py
# -*- coding: utf-8 -*-
"""
CCM / CSC with robust fallbacks + conservative blend (pure Python).

When DNG lacks matrices, provide three fallbacks:
  - bypass (default): return I₃
  - assume_xyz_d65:  Cam≈XYZ(D65) → XYZ→RGB(D65)
  - assume_xyz_d50:  Cam≈XYZ(D50) → CAT(D50→target) → XYZ→RGB(target)

Optional fallback_blend (0..1) blends toward the assumed matrix:
  M_final = (1 - blend) * I + blend * M_assume
Then row-normalize to reduce color cast risk.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    from tifffile import TiffFile
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False


# ---------- Basic utilities: parse SRATIONAL / matrices ----------
def _as_float_array(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=object)

    if a.size == 9 and a.ndim == 1:
        try:
            return a.astype(np.float64)
        except Exception:
            pass

    if a.size == 18 and a.ndim == 1:  # num,den, num,den, ...
        nums = a[0::2].astype(np.float64)
        dens = a[1::2].astype(np.float64)
        dens = np.where(dens == 0, 1.0, dens)
        return nums / dens

    if a.ndim == 2 and a.shape == (9, 2):
        nums = a[:, 0].astype(np.float64)
        dens = a[:, 1].astype(np.float64)
        dens = np.where(dens == 0, 1.0, dens)
        return nums / dens

    if a.size == 9 and a.ndim == 1:
        vals = []
        for v in a:
            if isinstance(v, (tuple, list)) and len(v) == 2:
                n, d = float(v[0]), float(v[1])
            elif hasattr(v, "numerator") and hasattr(v, "denominator"):
                n, d = float(v.numerator), float(v.denominator)
            else:
                try:
                    vals.append(float(v))
                    continue
                except Exception:
                    raise ValueError(f"Unsupported rational element type: {type(v)}")
            d = 1.0 if d == 0 else d
            vals.append(n / d)
        return np.array(vals, dtype=np.float64)

    try:
        af = a.astype(np.float64)
        if af.size == 9:
            return af
    except Exception:
        pass

    raise ValueError(f"Cannot parse SRATIONAL/Matrix with shape {a.shape} and size {a.size}")


def _to33(x):
    arr = _as_float_array(x)
    if arr is None:
        return None
    if arr.size < 9:
        raise ValueError(f"Matrix has insufficient elements: {arr.size}")
    return arr[:9].reshape(3, 3)


# ---------- DNG tag reading ----------
def _get_tag_any_page(tf: "TiffFile", tagname: str):
    for pg in tf.pages:
        if tagname in pg.tags:
            return pg.tags[tagname].value
    return None


def read_dng_mats(path: str) -> Dict[str, Optional[np.ndarray]]:
    """Read common DNG color-related tags; return as 3x3 arrays or None."""
    if not _HAS_TIFFFILE:
        return {}
    with TiffFile(path) as tf:
        def get(t): return _get_tag_any_page(tf, t)
        ret = {
            'ColorMatrix1':         _to33(get('ColorMatrix1'))  if get('ColorMatrix1')  is not None else None,
            'ColorMatrix2':         _to33(get('ColorMatrix2'))  if get('ColorMatrix2')  is not None else None,
            'CameraCalibration1':   _to33(get('CameraCalibration1')) if get('CameraCalibration1') is not None else None,
            'CameraCalibration2':   _to33(get('CameraCalibration2')) if get('CameraCalibration2') is not None else None,
            'ForwardMatrix1':       _to33(get('ForwardMatrix1')) if get('ForwardMatrix1') is not None else None,
            'ForwardMatrix2':       _to33(get('ForwardMatrix2')) if get('ForwardMatrix2') is not None else None,
            'ReductionMatrix1':     _to33(get('ReductionMatrix1')) if get('ReductionMatrix1') is not None else None,
            'ReductionMatrix2':     _to33(get('ReductionMatrix2')) if get('ReductionMatrix2') is not None else None,
            'CalibrationIlluminant1': get('CalibrationIlluminant1'),
            'CalibrationIlluminant2': get('CalibrationIlluminant2'),
            'AsShotNeutral':        get('AsShotNeutral'),
            'AsShotWhiteXY':        get('AsShotWhiteXY'),
            'AnalogBalance':        get('AnalogBalance'),
        }
    return ret


# ---------- Color space basics ----------
def _rgb2xyz_matrix(color_space: str = "srgb", illuminant: str = "d65") -> np.ndarray:
    """Return RGB→XYZ matrix for (sRGB|AdobeRGB1998) × (D65|D50)."""
    cs = (color_space or "srgb").lower()
    ill = (illuminant or "d65").lower()

    if cs == "srgb":
        if ill == "d65":
            return np.array([[0.4124564, 0.3575761, 0.1804375],
                             [0.2126729, 0.7151522, 0.0721750],
                             [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
        elif ill == "d50":
            return np.array([[0.4360747, 0.3850649, 0.1430804],
                             [0.2225045, 0.7168786, 0.0606169],
                             [0.0139322, 0.0971045, 0.7141733]], dtype=np.float64)

    elif cs == "adobe-rgb-1998":
        if ill == "d65":
            return np.array([[0.5767309, 0.1855540, 0.1881852],
                             [0.2973769, 0.6273491, 0.0752741],
                             [0.0270343, 0.0706872, 0.9911085]], dtype=np.float64)
        elif ill == "d50":
            return np.array([[0.6097559, 0.2052401, 0.1492240],
                             [0.3111242, 0.6256560, 0.0632197],
                             [0.0194811, 0.0608902, 0.7448387]], dtype=np.float64)

    raise ValueError("Unsupported (color_space, illuminant); supports (srgb|adobe-rgb-1998) × (d65|d50).")


# Bradford CAT
def _bradford() -> np.ndarray:
    return np.array([[ 0.8951,  0.2664, -0.1614],
                     [-0.7502,  1.7135,  0.0367],
                     [ 0.0389, -0.0685,  1.0296]], dtype=np.float64)

def _cat_matrix(src_white_xyz: np.ndarray, dst_white_xyz: np.ndarray) -> np.ndarray:
    """Compute Bradford CAT from src white to dst white."""
    M = _bradford()
    Minv = np.linalg.inv(M)
    src = M @ src_white_xyz
    dst = M @ dst_white_xyz
    D = np.diag(dst / np.where(src == 0, 1.0, src))
    return Minv @ D @ M

_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_WHITE_D50 = np.array([0.96422, 1.0, 0.82521], dtype=np.float64)


# ---------- Numerical robustness ----------
def _safe_inverse_3x3(m: np.ndarray) -> np.ndarray:
    """Inverse with condition guard; fall back to pinv if ill-conditioned."""
    cond = np.linalg.cond(m)
    try:
        if np.isfinite(cond) and cond < 1e12:
            return np.linalg.inv(m)
        return np.linalg.pinv(m)
    except Exception:
        return np.eye(3, dtype=np.float64)


# ---------- Lights & interpolation ----------
_LIGHTSOURCE_CCT = {
    9: 5500, 10: 6500, 11: 7500, 12: 6500, 13: 5100, 14: 4200, 15: 3450,
    17: 2850, 18: 5500, 19: 6500, 20: 5500, 21: 6504, 22: 7500, 23: 5400, 24: 5000,
}

def _cct_from_light_source(code: Any, default: float) -> float:
    """Map DNG light source code to rough CCT; fall back to default."""
    try:
        code = int(code)
        v = _LIGHTSOURCE_CCT.get(code, None)
        if v and v > 0:
            return float(v)
    except Exception:
        pass
    return float(default)

def _cct_from_xy(x: float, y: float) -> float:
    """McCamy-style CCT from chromaticity (clamped to 1500..20000K)."""
    y = 1e-12 if abs(y - 0.1858) < 1e-12 else y
    n = (x - 0.3320) / (y - 0.1858)
    cct = 449.0 * (n ** 3) + 3525.0 * (n ** 2) + 6823.3 * n + 5520.33
    return float(np.clip(cct, 1500.0, 20000.0))

def _mired(t_kelvin: float) -> float:
    """Convert kelvin to mireds."""
    t = max(t_kelvin, 1e-6)
    return 1e6 / t

def _interp_weight_mired(t1: float, t2: float, ts: float) -> float:
    """Linear interpolation in mired domain to get weight for the second matrix."""
    m1, m2, ms = _mired(t1), _mired(t2), _mired(ts)
    if abs(m2 - m1) < 1e-9:
        return 0.5
    alpha = (ms - m1) / (m2 - m1)
    return float(np.clip(alpha, 0.0, 1.0))


# ---------- Compose Cam→XYZ (interpolation + priority) ----------
def _compose_cam2xyz_interpolated(mats: Dict[str, Optional[np.ndarray]],
                                  info: Dict[str, Any]) -> Optional[np.ndarray]:
    """Prefer ForwardMatrix (with CC) else ColorMatrix (with CC); interpolate by CCT."""
    if not mats:
        return None

    FM1, FM2 = mats.get('ForwardMatrix1'), mats.get('ForwardMatrix2')
    CC1, CC2 = mats.get('CameraCalibration1'), mats.get('CameraCalibration2')
    CM1, CM2 = mats.get('ColorMatrix1'), mats.get('ColorMatrix2')
    ill1, ill2 = mats.get('CalibrationIlluminant1'), mats.get('CalibrationIlluminant2')

    shot_cct = None
    if mats.get('AsShotWhiteXY') is not None:
        try:
            x, y = mats['AsShotWhiteXY']
            shot_cct = _cct_from_xy(float(x), float(y))
            info["as_shot_white_xy"] = [float(x), float(y)]
            info["shot_cct_from_xy"] = shot_cct
        except Exception:
            pass

    cct1 = _cct_from_light_source(ill1, 2850.0)
    cct2 = _cct_from_light_source(ill2, 6500.0)
    if shot_cct is None:
        shot_cct = cct2 if (abs(cct2 - 6500) < abs(cct1 - 6500)) else (0.5 * (cct1 + cct2))
        info.setdefault("warnings", []).append("AsShotWhiteXY missing -> shot_cct estimated")

    alpha = _interp_weight_mired(cct1, cct2, shot_cct)
    info["interpolation"] = {"cct1": cct1, "cct2": cct2, "shot_cct": shot_cct, "alpha_for_#2": alpha}

    def _apply_cc(cam2xyz: np.ndarray, cc: Optional[np.ndarray]) -> np.ndarray:
        return cam2xyz if cc is None else cam2xyz @ _safe_inverse_3x3(cc.astype(np.float64))

    if FM1 is not None or FM2 is not None:
        used, parts = [], []
        if FM1 is not None:
            parts.append(_apply_cc(FM1.astype(np.float64), CC1)); used.append("ForwardMatrix1")
        if FM2 is not None:
            parts.append(_apply_cc(FM2.astype(np.float64), CC2)); used.append("ForwardMatrix2")
        info["used_tags"] = used
        if len(parts) == 1:
            return parts[0]
        return (1.0 - alpha) * parts[0] + alpha * parts[1]

    if (CM1 is not None) or (CM2 is not None):
        used, parts = [], []
        if CM1 is not None:
            xyz2cam1 = CM1.astype(np.float64)
            if CC1 is not None: xyz2cam1 = CC1.astype(np.float64) @ xyz2cam1
            parts.append(_safe_inverse_3x3(xyz2cam1)); used.append("ColorMatrix1(+CC1)")
        if CM2 is not None:
            xyz2cam2 = CM2.astype(np.float64)
            if CC2 is not None: xyz2cam2 = CC2.astype(np.float64) @ xyz2cam2
            parts.append(_safe_inverse_3x3(xyz2cam2)); used.append("ColorMatrix2(+CC2)")
        info["used_tags"] = used
        if len(parts) == 1:
            return parts[0]
        return (1.0 - alpha) * parts[0] + alpha * parts[1]

    return None


# ---------- Compatibility: legacy xyz2cam-only path ----------
def _compose_xyz2cam(mats: Dict[str, Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Return xyz2cam from ColorMatrix2(+CC2) or inverse of FM2 when present."""
    if not mats:
        return None
    cm2 = mats.get('ColorMatrix2')
    cc2 = mats.get('CameraCalibration2')
    fm2 = mats.get('ForwardMatrix2')

    if cm2 is not None:
        xyz2cam = cm2.astype(np.float64)
        if cc2 is not None:
            xyz2cam = cc2.astype(np.float64) @ xyz2cam
        return xyz2cam

    if fm2 is not None:
        try:
            cam2xyz = fm2.astype(np.float64)
            return np.linalg.inv(cam2xyz)
        except Exception:
            return None
    return None


# ---------- Compute Cam→RGB (with CSC/CAT, row normalization) ----------
def calculate_cam2rgb_from_meta(color_space: str,
                                illuminant: str,
                                mats_from_dng: Optional[Dict[str, Optional[np.ndarray]]] = None,
                                xyz2cam_from_meta: Optional[np.ndarray] = None,
                                require_camera_mats: bool = False,
                                fallback_strategy: str = "bypass",     # 'bypass' | 'assume_xyz_d65' | 'assume_xyz_d50'
                                fallback_blend: float = 0.15            # blend toward assumed matrix in fallback
                                ) -> Tuple[np.ndarray, Dict[str, Any]]:

    rgb2xyz = _rgb2xyz_matrix(color_space, illuminant)
    xyz2rgb = _safe_inverse_3x3(rgb2xyz.astype(np.float64))
    info: Dict[str, Any] = {"color_space": color_space, "illuminant": illuminant, "warnings": []}

    # 1) Try cam→XYZ(D50) from DNG/meta (ForwardMatrix preferred; else CM+CC).
    cam2xyz = None
    if mats_from_dng:
        cam2xyz = _compose_cam2xyz_interpolated(mats_from_dng, info)

    if cam2xyz is None:
        xyz2cam = None
        if xyz2cam_from_meta is not None:
            xyz2cam = np.asarray(xyz2cam_from_meta, dtype=np.float64)
            info["used_tags"] = ["xyz2cam_from_meta"]
        elif mats_from_dng:
            xyz2cam = _compose_xyz2cam(mats_from_dng)
            if xyz2cam is not None:
                info.setdefault("warnings", []).append("Using xyz2cam (no ForwardMatrix/CM1/2 interp).")
                info["used_tags"] = ["ColorMatrix2(+CC2) or FM2^-1"]
        if xyz2cam is not None:
            cam2xyz = _safe_inverse_3x3(xyz2cam)

    # 2) Fallbacks when no matrix is available.
    if cam2xyz is None:
        if require_camera_mats:
            raise RuntimeError(
                "No camera matrices found. Provide ForwardMatrix1/2 or ColorMatrix1/2(+CameraCalibration), "
                "or pass a 3x3 ColorMatrix2 via meta."
            )
        used = "fallback_" + str(fallback_strategy).lower()
        info["used_tags"] = [used]

        if str(fallback_strategy).lower() == "bypass":
            cam2rgb64 = np.eye(3, dtype=np.float64)
            info.setdefault("warnings", []).append("Fallback: BYPASS (I3).")
            # Row-normalize (no-op for I3).
            row_sums = np.sum(cam2rgb64, axis=1, keepdims=True)
            cam2rgb64 = cam2rgb64 / np.where(np.abs(row_sums) < 1e-12, 1.0, row_sums)

            info.update({
                "rgb2xyz": rgb2xyz.tolist(),
                "xyz2rgb": xyz2rgb.tolist(),
                "xyz_cat": np.eye(3, dtype=np.float64).tolist(),
                "cam2rgb": cam2rgb64.tolist(),
                "cond_cam2rgb": float(np.linalg.cond(cam2rgb64)),
                "fallback_blend": 0.0
            })
            return cam2rgb64.astype(np.float32), info

        # Build assumed matrix.
        if str(fallback_strategy).lower() == "assume_xyz_d65":
            # Cam≈XYZ(D65) → XYZ→RGB(D65).
            M_assume = xyz2rgb.copy()
            info.setdefault("warnings", []).append(
                "Fallback: assume Cam≈XYZ(D65) (blend towards XYZ→RGB)."
            )
            xyz_cat = np.eye(3, dtype=np.float64)

        elif str(fallback_strategy).lower() == "assume_xyz_d50":
            # Cam≈XYZ(D50) → CAT(D50→target) → XYZ→RGB(target).
            target_ill = (illuminant or "d65").lower()
            if target_ill == "d65":
                xyz_cat = _cat_matrix(_WHITE_D50, _WHITE_D65)
            elif target_ill == "d50":
                xyz_cat = np.eye(3, dtype=np.float64)
            else:
                xyz_cat = np.eye(3, dtype=np.float64)
                info.setdefault("warnings", []).append(f"Unknown target illuminant '{illuminant}', skip CAT.")
            M_assume = xyz2rgb @ xyz_cat
            info.setdefault("warnings", []).append(
                "Fallback: assume Cam≈XYZ(D50) (blend towards CAT*XYZ→RGB)."
            )
        else:
            # Unknown fallback → bypass.
            M_assume = np.eye(3, dtype=np.float64)
            xyz_cat = np.eye(3, dtype=np.float64)
            info.setdefault("warnings", []).append(f"Unknown fallback '{fallback_strategy}', BYPASS used.")

        # Conservative blend with identity, then row-normalize.
        t = float(np.clip(fallback_blend, 0.0, 1.0))
        cam2rgb64 = (1.0 - t) * np.eye(3, dtype=np.float64) + t * M_assume
        row_sums = np.sum(cam2rgb64, axis=1, keepdims=True)
        cam2rgb64 = cam2rgb64 / np.where(np.abs(row_sums) < 1e-12, 1.0, row_sums)

        info.update({
            "rgb2xyz": rgb2xyz.tolist(),
            "xyz2rgb": xyz2rgb.tolist(),
            "xyz_cat": xyz_cat.tolist(),
            "cam2rgb": cam2rgb64.tolist(),
            "cond_cam2rgb": float(np.linalg.cond(cam2rgb64)),
            "fallback_blend": t
        })
        return cam2rgb64.astype(np.float32), info

    # 3) Normal path: assume cam2xyz is D50-referenced; CAT to target and map to RGB.
    info["cam2xyz_assumed_white"] = "D50"
    target_ill = (illuminant or "d65").lower()
    if target_ill == "d65":
        xyz_cat = _cat_matrix(_WHITE_D50, _WHITE_D65)
    elif target_ill == "d50":
        xyz_cat = np.eye(3, dtype=np.float64)
    else:
        xyz_cat = np.eye(3, dtype=np.float64)
        info.setdefault("warnings", []).append(f"Unknown target illuminant '{illuminant}', skip CAT.")

    cam2rgb64 = xyz2rgb @ (xyz_cat @ cam2xyz)

    # Row-normalize to preserve neutral axis.
    row_sums = np.sum(cam2rgb64, axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) < 1e-12, 1.0, row_sums)
    cam2rgb64 = cam2rgb64 / row_sums

    info.update({
        "rgb2xyz": rgb2xyz.tolist(),
        "xyz2rgb": xyz2rgb.tolist(),
        "xyz_cat": xyz_cat.tolist(),
        "cam2rgb": cam2rgb64.tolist(),
        "cond_cam2rgb": float(np.linalg.cond(cam2rgb64)),
        "fallback_blend": 1.0
    })
    return cam2rgb64.astype(np.float32), info


# ---------- Apply CCM (Cam→RGB) ----------
def apply_ccm_to_rgb(rgb_lin: np.ndarray,
                     cam2rgb: np.ndarray,
                     clip_min: float,
                     clip_max: float) -> np.ndarray:
    """Apply 3×3 CCM to H×W×3 linear RGB and clip to range."""
    if rgb_lin.ndim != 3 or rgb_lin.shape[2] != 3:
        raise ValueError(f"apply_ccm_to_rgb expects HxWx3, got {rgb_lin.shape}")
    M = np.asarray(cam2rgb, dtype=np.float32)
    out = np.tensordot(rgb_lin.astype(np.float32), M.T.astype(np.float32), axes=1)
    return np.clip(out, float(clip_min), float(clip_max))


def _apply_linear_look(rgb_lin: np.ndarray,
                       WL: float,
                       L: Optional[np.ndarray],
                       bias: Optional[np.ndarray],
                       *,
                       strength: float = 1.0,
                       preserve_neutral: bool = False) -> np.ndarray:
    """
    Apply a 3×3+bias “look” in linear RGB [0..WL].
    strength mixes with identity; optional row-normalization preserves neutrals.
    """
    if L is None and bias is None:
        return rgb_lin

    x = (rgb_lin.astype(np.float32) / float(WL)).clip(0.0, 1.0)
    I = np.eye(3, dtype=np.float32)

    if L is not None:
        L = np.asarray(L, dtype=np.float32).reshape(3,3)
        Leff = (1.0 - float(strength)) * I + float(strength) * L
        if preserve_neutral:
            rs = Leff.sum(axis=1, keepdims=True)
            rs = np.where(np.abs(rs) < 1e-12, 1.0, rs)
            Leff = Leff / rs
        x = np.tensordot(x, Leff.T, axes=1)

    if bias is not None:
        b = np.asarray(bias, dtype=np.float32).reshape(3)
        x = x + float(strength) * b

    x = np.clip(x, 0.0, 1.0)
    return x * float(WL)


# ---------- Public API ----------
def run_ccm_rgb(rgb_lin: np.ndarray,
                meta_in: Dict[str, Any],
                out_dir: str,
                color_space: str = "srgb",
                illuminant: str = "d65",
                save_json: bool = True,
                require_camera_mats: bool = False,
                fallback_strategy: str = "bypass",   # 'bypass' | 'assume_xyz_d65' | 'assume_xyz_d50'
                fallback_blend: float = 0.15,        # only used in assume_* fallbacks
                cheat_strength: float = 0.0,         # 0=identity, 1=use computed matrix

                post_look_matrix = None,
                post_look_bias = None,
                post_look_strength : float = 1.0,
                post_look_preserve_neutral: bool = False

                ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    cheat_strength shrinks cam2rgb toward identity (0..1).
      0.0 → I3 (almost no change); 0.05~0.10 → very mild correction; 1.0 → full cam2rgb.
    """
    dng_path = meta_in.get("file_path") or meta_in.get("dng_path")
    mats = None
    if dng_path and Path(dng_path).exists():
        try:
            mats = read_dng_mats(str(dng_path))
        except Exception:
            mats = None

    # Accept direct xyz2cam from meta for compatibility.
    xyz2cam_from_meta = None
    if "ColorMatrix2" in meta_in and isinstance(meta_in["ColorMatrix2"], (list, np.ndarray)):
        try:
            xyz2cam_from_meta = np.asarray(meta_in["ColorMatrix2"], dtype=np.float64).reshape(3, 3)
        except Exception:
            xyz2cam_from_meta = None

    # Compute nominal cam2rgb first.
    cam2rgb_nominal, info = calculate_cam2rgb_from_meta(
        color_space=color_space,
        illuminant=illuminant,
        mats_from_dng=mats,
        xyz2cam_from_meta=xyz2cam_from_meta,
        require_camera_mats=require_camera_mats,
        fallback_strategy=fallback_strategy,
        fallback_blend=fallback_blend,
    )

    # Shrink globally toward identity, then row-normalize.
    t = float(np.clip(cheat_strength, 0.0, 1.0))
    I = np.eye(3, dtype=np.float64)
    M_nom = cam2rgb_nominal.astype(np.float64)
    if t <= 1e-12:
        M_final = I.copy()
    else:
        M_final = (1.0 - t) * I + t * M_nom

    row_sums = np.sum(M_final, axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) < 1e-12, 1.0, row_sums)
    M_final = M_final / row_sums

    # Apply CCM.
    WL = float(meta_in.get("white_level", 1.0))
    rgb_ccm = apply_ccm_to_rgb(rgb_lin, M_final.astype(np.float32), 0.0, WL)

    # Optional post-look (3×3 + bias) in linear RGB.
    rgb_ccm = _apply_linear_look(rgb_ccm, WL,
                                 post_look_matrix,
                                 post_look_bias,
                                 strength=post_look_strength,
                                 preserve_neutral=post_look_preserve_neutral)

    # Metadata / info.
    meta_out = dict(meta_in)
    meta_out["ccm_color_space"] = color_space
    meta_out["ccm_illuminant"] = illuminant
    meta_out["cam2rgb"] = M_final.tolist()        # actual matrix used (after shrink)
    meta_out["fallback_strategy"] = fallback_strategy
    meta_out["fallback_blend"] = float(fallback_blend)
    meta_out["cheat_strength"] = t

    info = dict(info)
    info["cam2rgb_nominal"] = info.get("cam2rgb")
    info["cam2rgb_final"] = M_final.tolist()
    info["cheat_strength"] = t
    info["note"] = "cam2rgb_final = (1 - cheat_strength)*I + cheat_strength*cam2rgb_nominal, then row-normalized."

    if save_json:
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(out_dir) / "ccm_info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return rgb_ccm, meta_out, info
