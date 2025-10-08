# # core/cbm3d_yuv.py
# # -*- coding: utf-8 -*-
# """
# CBM3D（在 YCbCr/YUV444 域执行 BM3D）
# - 输入：Y、U、V 三平面（同尺寸），dtype 可为 uint8/uint16/float32
# - 处理：统一归一到 0..1；U/V 若为中心在 0（-0.5..0.5），则平移到 0..1 做 BM3D；
#        三通道各自 BM3D（Basic+Wiener，复用 core.bm3d 的“快版”内核）；
#        最后映射回原位深/范围。
# - 输出：去噪后的 Y、U、V，meta_out，info（含 σ 等）
# """
#
# import os, json, logging
# import numpy as np
#
# # 复用你已有 BM3D 内核与工具（来自 core/bm3d.py）
# from .bm3d import (
#     _BM3DParams,
#     _bm3d_gray_numpy_fast,
#     _robust_sigma_mad,
# )
#
# logger = logging.getLogger(__name__)
#
# # -------------------- I/O 与归一 --------------------
#
# def _infer_scale_from_dtype(arr: np.ndarray) -> float:
#     if np.issubdtype(arr.dtype, np.uint8):  return 255.0
#     if np.issubdtype(arr.dtype, np.uint16): return 65535.0
#     return 1.0
#
# def _to01(x):
#     s = _infer_scale_from_dtype(x)
#     return np.clip(np.asarray(x, np.float32) / max(s, 1.0), 0.0, 1.0), s
#
# def _from01(x01, scale_like: float, dtype_like):
#     y = np.clip(np.asarray(x01, np.float32), 0.0, 1.0) * float(scale_like)
#     if np.issubdtype(dtype_like, np.uint8):  return np.rint(y).astype(np.uint8)
#     if np.issubdtype(dtype_like, np.uint16): return np.rint(y).astype(np.uint16)
#     return y.astype(np.float32)
#
# def _save_u8_gray(path, img01):
#     try:
#         import imageio.v2 as iio
#         x = np.asarray(img01, np.float32)
#         if x.size == 0: return
#         lo = float(np.quantile(x, 0.01)); hi = float(np.quantile(x, 0.999))
#         if not np.isfinite(lo): lo = 0.0
#         if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
#         x = np.clip((x - lo) / (hi - lo), 0, 1)
#         iio.imwrite(str(path), np.rint(x*255.0).astype(np.uint8))
#     except Exception as e:
#         logger.debug("[CBM3D] save png skipped: %s", e)
#
# def _save_planar_yuv(path_stem: str, Y, U, V):
#     try:
#         with open(path_stem + ".yuv", "wb") as f:
#             f.write(np.ascontiguousarray(Y).tobytes())
#             f.write(np.ascontiguousarray(U).tobytes())
#             f.write(np.ascontiguousarray(V).tobytes())
#     except Exception as e:
#         logger.debug("[CBM3D] save .yuv skipped: %s", e)
#
# def _center_uv_if_needed(U01, V01):
#     """判断 U/V 中性：0.5（0..1）还是 0（-0.5..0.5）；必要时平移到 0..1."""
#     mu_u, mu_v = float(np.mean(U01)), float(np.mean(V01))
#     if 0.35 <= mu_u <= 0.65 and 0.35 <= mu_v <= 0.65:
#         def inv(u01, v01): return u01, v01
#         return U01, V01, inv
#     Uc = np.clip(U01 + 0.5, 0.0, 1.0)
#     Vc = np.clip(V01 + 0.5, 0.0, 1.0)
#     def inv(u01, v01): return (u01 - 0.5, v01 - 0.5)
#     return Uc, Vc, inv
#
# # -------------------- 核心：CBM3D on YUV444 --------------------
#
# def run_cbm3d_yuv444(yuv_in, meta_in: dict, out_dir: str,
#                      *,
#                      sigma=None,                 # None/标量/字典（'Y','U','V'），单位：0..1
#                      strength: float = 1.0,      # 全局强度
#                      chroma_sigma_scale: float = 1.2,  # U/V 额外放大
#                      sigma_cap01: float = 0.25,  # σ 上限
#                      sigma_floor01: float = 0.0, # σ 下限
#                      save_plots: bool = True,
#                      save_yuv: bool = True,
#                      filename_stem: str = "cbm3d_yuv444"):
#     """
#     在 YUV444 域执行 BM3D（Basic+Wiener）
#     返回：(Y_dn, U_dn, V_dn), meta_out, info
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     meta = dict(meta_in or {})
#
#     # 拆三平面
#     if isinstance(yuv_in, (tuple, list)) and len(yuv_in) == 3:
#         Y_in, U_in, V_in = yuv_in
#     else:
#         arr = np.asarray(yuv_in)
#         assert arr.ndim == 3 and arr.shape[2] == 3, "yuv_in 必须是 (Y,U,V) 或 HxWx3"
#         Y_in, U_in, V_in = arr[...,0], arr[...,1], arr[...,2]
#
#     H, W = Y_in.shape[:2]
#     assert U_in.shape[:2] == (H, W) and V_in.shape[:2] == (H, W), "U/V 尺寸必须与 Y 相同"
#
#     Y01, sY = _to01(Y_in); U01, sU = _to01(U_in); V01, sV = _to01(V_in)
#
#     # U/V 范围处理
#     U01_for_dn, V01_for_dn, inv_uv = _center_uv_if_needed(U01, V01)
#
#     # 估噪（归一域）
#     if sigma is None:
#         sY_est = _robust_sigma_mad(np.asarray(Y01, np.float32))
#         Uc = U01 - 0.5; Vc = V01 - 0.5
#         sU_est = _robust_sigma_mad(np.asarray(Uc, np.float32))
#         sV_est = _robust_sigma_mad(np.asarray(Vc, np.float32))
#         sigma_map = {"Y": sY_est, "U": sU_est, "V": sV_est}
#     elif np.isscalar(sigma):
#         s = float(sigma); sigma_map = {"Y": s, "U": s, "V": s}
#     elif isinstance(sigma, dict):
#         sigma_map = {"Y": float(sigma.get("Y", 0.0)),
#                      "U": float(sigma.get("U", 0.0)),
#                      "V": float(sigma.get("V", 0.0))}
#     else:
#         raise ValueError("sigma 必须是 None/标量/字典（键：'Y','U','V'），单位为 0..1 归一域")
#
#     def _cap(s):
#         s = max(float(s) * float(strength), float(sigma_floor01))
#         return max(min(s, float(sigma_cap01)), 1e-4)
#
#     sY_used  = _cap(sigma_map["Y"])
#     sU_used  = _cap(sigma_map["U"] * float(chroma_sigma_scale))
#     sV_used  = _cap(sigma_map["V"] * float(chroma_sigma_scale))
#     sigma_used = {"Y": sY_used, "U": sU_used, "V": sV_used}
#
#     # 逐通道 BM3D（复用快版内核）
#     def _denoise01(ch01, s01):
#         den01, _ = _bm3d_gray_numpy_fast(ch01, sigma=s01, p=_BM3DParams(
#             patch_size=8, step=8, search_window=17, max_group_size=12,
#             hard_thr_lambda=2.9, match_tau_mult=900.0, kaiser_beta=2.0
#         ))
#         return den01
#
#     Y_dn01 = _denoise01(np.clip(Y01, 0.0, 1.0), sY_used)
#     U_dn01 = _denoise01(np.clip(U01_for_dn, 0.0, 1.0), sU_used)
#     V_dn01 = _denoise01(np.clip(V01_for_dn, 0.0, 1.0), sV_used)
#
#     # U/V 逆映射回原“中心”
#     U_dn01, V_dn01 = inv_uv(U_dn01, V_dn01)
#
#     # 回写原位深 dtype
#     Y_dn = _from01(Y_dn01, sY, Y_in.dtype)
#     U_dn = _from01(U_dn01, sU, U_in.dtype)
#     V_dn = _from01(V_dn01, sV, V_in.dtype)
#
#     # 保存
#     stem = os.path.splitext(os.path.basename(meta.get("file_basename", "image")))[0]
#     if save_plots:
#         _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_Y_in.png"),  Y01)
#         _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_U_in.png"),  U01)
#         _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_V_in.png"),  V01)
#         _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_Y_out.png"), Y_dn01)
#         _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_U_out.png"), np.clip(U_dn01 + 0.5, 0.0, 1.0))
#         _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_V_out.png"), np.clip(V_dn01 + 0.5, 0.0, 1.0))
#         try:
#             with open(os.path.join(out_dir, f"{stem}_CBM3D_info.json"), "w", encoding="utf-8") as f:
#                 json.dump({
#                     "sigma_est": {k: float(sigma_map[k]) for k in ("Y","U","V")},
#                     "sigma_used": {k: float(sigma_used[k]) for k in ("Y","U","V")},
#                     "strength": float(strength),
#                     "chroma_sigma_scale": float(chroma_sigma_scale),
#                     "sigma_cap01": float(sigma_cap01),
#                     "sigma_floor01": float(sigma_floor01),
#                     "shape": [int(H), int(W)],
#                     "dtype_in": {"Y": str(Y_in.dtype), "U": str(U_in.dtype), "V": str(V_in.dtype)},
#                 }, f, ensure_ascii=False, indent=2)
#         except Exception:
#             pass
#
#     if save_yuv:
#         _save_planar_yuv(os.path.join(out_dir, f"{stem}_CBM3D_out"), Y_dn, U_dn, V_dn)
#
#     # meta / info
#     meta_out = dict(meta)
#     meta_out.setdefault("denoise", {})
#     meta_out["denoise"].update({
#         "method": "cbm3d_numpy_fast",
#         "sigma_cap01": float(sigma_cap01),
#         "sigma_floor01": float(sigma_floor01),
#         "strength": float(strength),
#         "chroma_sigma_scale": float(chroma_sigma_scale),
#         "colorspace": "YCbCr(YUV444)"
#     })
#     info = {"backend": "cbm3d_numpy_fast", "sigma_used": sigma_used, "shape": [int(H), int(W)]}
#
#     logger.info("[CBM3D][YUV444] σ(Y/U/V)=%s saved_plots=%s",
#                 {k: round(v, 4) for k, v in sigma_used.items()},
#                 str(bool(save_plots)))
#     return (Y_dn, U_dn, V_dn), meta_out, info


# core/cbm3d_yuv.py
# -*- coding: utf-8 -*-
"""
CBM3D（在 YCbCr/YUV444 域执行 BM3D）
- 输入：Y、U、V 三平面（同尺寸），dtype 可为 uint8/uint16/float32
- 处理：统一归一到 0..1；U/V 若为中心在 0（-0.5..0.5），则平移到 0..1 做 BM3D；
       三通道各自 BM3D（Basic+Wiener，复用 core.bm3d 的“快版”内核）；
       最后映射回原位深/范围，并保存灰度/彩色预览。
- 输出：去噪后的 Y、U、V，meta_out，info（含 σ 等）
"""

import os, json, logging
import numpy as np

# 复用你已有 BM3D 内核与工具（来自 core/bm3d.py）
from core.bm3d import (
    _BM3DParams,
    _bm3d_gray_numpy_fast,
    _robust_sigma_mad,
)

logger = logging.getLogger(__name__)

# -------------------- I/O 与归一 --------------------

def _infer_scale_from_dtype(arr: np.ndarray) -> float:
    if np.issubdtype(arr.dtype, np.uint8):  return 255.0
    if np.issubdtype(arr.dtype, np.uint16): return 65535.0
    return 1.0

def _to01(x):
    s = _infer_scale_from_dtype(x)
    return np.clip(np.asarray(x, np.float32) / max(s, 1.0), 0.0, 1.0), s

def _from01(x01, scale_like: float, dtype_like):
    y = np.clip(np.asarray(x01, np.float32), 0.0, 1.0) * float(scale_like)
    if np.issubdtype(dtype_like, np.uint8):  return np.rint(y).astype(np.uint8)
    if np.issubdtype(dtype_like, np.uint16): return np.rint(y).astype(np.uint16)
    return y.astype(np.float32)

def _save_u8_gray(path, img01):
    try:
        import imageio.v2 as iio
        x = np.asarray(img01, np.float32)
        if x.size == 0: return
        lo = float(np.quantile(x, 0.01)); hi = float(np.quantile(x, 0.999))
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        x = np.clip((x - lo) / (hi - lo), 0, 1)
        iio.imwrite(str(path), np.rint(x*255.0).astype(np.uint8))
    except Exception as e:
        logger.debug("[CBM3D] save png skipped: %s", e)

def _save_planar_yuv(path_stem: str, Y, U, V):
    try:
        with open(path_stem + ".yuv", "wb") as f:
            f.write(np.ascontiguousarray(Y).tobytes())
            f.write(np.ascontiguousarray(U).tobytes())
            f.write(np.ascontiguousarray(V).tobytes())
    except Exception as e:
        logger.debug("[CBM3D] save .yuv skipped: %s", e)

def _center_uv_if_needed(U01, V01):
    """判断 U/V 中性：0.5（0..1）还是 0（-0.5..0.5）；必要时平移到 0..1."""
    mu_u, mu_v = float(np.mean(U01)), float(np.mean(V01))
    if 0.35 <= mu_u <= 0.65 and 0.35 <= mu_v <= 0.65:
        def inv(u01, v01): return u01, v01
        return U01, V01, inv
    Uc = np.clip(U01 + 0.5, 0.0, 1.0)
    Vc = np.clip(V01 + 0.5, 0.0, 1.0)
    def inv(u01, v01): return (u01 - 0.5, v01 - 0.5)
    return Uc, Vc, inv

# -------------------- 彩色预览：YUV444 -> RGB(sRGB) --------------------

def _yuv444_to_rgb01_bt601(Y01, Cb_zero_center, Cr_zero_center, clamp=True):
    """
    输入：
      - Y01: 0..1
      - Cb_zero_center, Cr_zero_center: 零中心（≈[-0.5,0.5]）
    输出：
      - RGB 0..1（线性）
    """
    Kr, Kb = 0.299, 0.114
    Kg = 1.0 - Kr - Kb
    Y = np.asarray(Y01, np.float32)
    Cb = np.asarray(Cb_zero_center, np.float32)
    Cr = np.asarray(Cr_zero_center, np.float32)

    R = Y + Cr * (2.0 * (1.0 - Kr))
    B = Y + Cb * (2.0 * (1.0 - Kb))
    G = (Y - Kr * R - Kb * B) / Kg

    if clamp:
        R = np.clip(R, 0.0, 1.0)
        G = np.clip(G, 0.0, 1.0)
        B = np.clip(B, 0.0, 1.0)

    return np.stack([R, G, B], axis=-1).astype(np.float32)

def _srgb_oetf(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01, 0.0, 1.0).astype(np.float32)
    a = 0.055; thr = 0.0031308
    y = np.where(x <= thr, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)
    return np.clip(y, 0.0, 1.0)

def _save_color_png(path, rgb01):
    try:
        import imageio.v2 as iio
        srgb = _srgb_oetf(np.asarray(rgb01, np.float32))
        iio.imwrite(str(path), np.rint(np.clip(srgb, 0, 1) * 255.0).astype(np.uint8))
    except Exception as e:
        logger.debug("[CBM3D] save color png skipped: %s", e)

# -------------------- 核心：CBM3D on YUV444 --------------------

def run_cbm3d_yuv444(yuv_in, meta_in: dict, out_dir: str,
                     *,
                     sigma=None,                 # None/标量/字典（'Y','U','V'），单位：0..1
                     strength: float = 1.0,      # 全局强度
                     chroma_sigma_scale: float = 1.2,  # U/V 额外放大
                     sigma_cap01: float = 0.25,  # σ 上限
                     sigma_floor01: float = 0.0, # σ 下限
                     save_plots: bool = True,
                     save_yuv: bool = True,
                     save_color: bool = True,    # <--- 新增：保存彩色图
                     filename_stem: str = "cbm3d_yuv444"):
    """
    在 YUV444 域执行 BM3D（Basic+Wiener）
    返回：(Y_dn, U_dn, V_dn), meta_out, info
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(meta_in or {})

    # 拆三平面
    if isinstance(yuv_in, (tuple, list)) and len(yuv_in) == 3:
        Y_in, U_in, V_in = yuv_in
    else:
        arr = np.asarray(yuv_in)
        assert arr.ndim == 3 and arr.shape[2] == 3, "yuv_in 必须是 (Y,U,V) 或 HxWx3"
        Y_in, U_in, V_in = arr[...,0], arr[...,1], arr[...,2]

    H, W = Y_in.shape[:2]
    assert U_in.shape[:2] == (H, W) and V_in.shape[:2] == (H, W), "U/V 尺寸必须与 Y 相同"

    Y01, sY = _to01(Y_in); U01, sU = _to01(U_in); V01, sV = _to01(V_in)

    # U/V 范围处理
    U01_for_dn, V01_for_dn, inv_uv = _center_uv_if_needed(U01, V01)

    # 估噪（归一域）
    if sigma is None:
        sY_est = _robust_sigma_mad(np.asarray(Y01, np.float32))
        Uc = U01 - 0.5; Vc = V01 - 0.5
        sU_est = _robust_sigma_mad(np.asarray(Uc, np.float32))
        sV_est = _robust_sigma_mad(np.asarray(Vc, np.float32))
        sigma_map = {"Y": sY_est, "U": sU_est, "V": sV_est}
    elif np.isscalar(sigma):
        s = float(sigma); sigma_map = {"Y": s, "U": s, "V": s}
    elif isinstance(sigma, dict):
        sigma_map = {"Y": float(sigma.get("Y", 0.0)),
                     "U": float(sigma.get("U", 0.0)),
                     "V": float(sigma.get("V", 0.0))}
    else:
        raise ValueError("sigma 必须是 None/标量/字典（键：'Y','U','V'），单位为 0..1 归一域")

    def _cap(s):
        s = max(float(s) * float(strength), float(sigma_floor01))
        return max(min(s, float(sigma_cap01)), 1e-4)

    sY_used  = _cap(sigma_map["Y"])
    sU_used  = _cap(sigma_map["U"] * float(chroma_sigma_scale))
    sV_used  = _cap(sigma_map["V"] * float(chroma_sigma_scale))
    sigma_used = {"Y": sY_used, "U": sU_used, "V": sV_used}

    # 逐通道 BM3D（复用快版内核）
    def _denoise01(ch01, s01):
        den01, _ = _bm3d_gray_numpy_fast(ch01, sigma=s01, p=_BM3DParams(
            patch_size=8, step=8, search_window=17, max_group_size=12,
            hard_thr_lambda=2.9, match_tau_mult=900.0, kaiser_beta=2.0
        ))
        return den01

    Y_dn01 = _denoise01(np.clip(Y01, 0.0, 1.0), sY_used)
    U_dn01 = _denoise01(np.clip(U01_for_dn, 0.0, 1.0), sU_used)
    V_dn01 = _denoise01(np.clip(V01_for_dn, 0.0, 1.0), sV_used)

    # U/V 逆映射回原“中心”
    U_dn01, V_dn01 = inv_uv(U_dn01, V_dn01)

    # 回写原位深 dtype
    Y_dn = _from01(Y_dn01, sY, Y_in.dtype)
    U_dn = _from01(U_dn01, sU, U_in.dtype)
    V_dn = _from01(V_dn01, sV, V_in.dtype)

    # ===== 保存灰度/信息 =====
    stem = os.path.splitext(os.path.basename(meta.get("file_basename", "image")))[0]
    if save_plots:
        _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_Y_in.png"),  Y01)
        _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_U_in.png"),  U01)
        _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_V_in.png"),  V01)
        _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_Y_out.png"), Y_dn01)
        # 用 +0.5 只是为了更直观可视化色度
        _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_U_out.png"), np.clip(U_dn01 + 0.5, 0.0, 1.0))
        _save_u8_gray(os.path.join(out_dir, f"{stem}_CBM3D_V_out.png"), np.clip(V_dn01 + 0.5, 0.0, 1.0))
        try:
            with open(os.path.join(out_dir, f"{stem}_CBM3D_info.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "sigma_est": {k: float(sigma_map[k]) for k in ("Y","U","V")},
                    "sigma_used": {k: float(sigma_used[k]) for k in ("Y","U","V")},
                    "strength": float(strength),
                    "chroma_sigma_scale": float(chroma_sigma_scale),
                    "sigma_cap01": float(sigma_cap01),
                    "sigma_floor01": float(sigma_floor01),
                    "shape": [int(H), int(W)],
                    "dtype_in": {"Y": str(Y_in.dtype), "U": str(U_in.dtype), "V": str(V_in.dtype)},
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    if save_yuv:
        _save_planar_yuv(os.path.join(out_dir, f"{stem}_CBM3D_out"), Y_dn, U_dn, V_dn)

    # ===== 新增：保存彩色预览 =====
    if save_color:
        # 先把（可能为 8/16bit）回到 0..1，得到与 Y 一致的归一域
        Yp01, _ = _to01(Y_dn)
        Up01, _ = _to01(U_dn)
        Vp01, _ = _to01(V_dn)
        # 判断 U/V 的“中心”：若均在 [0.35,0.65] 视为 0..1，需要减 0.5；否则视为零中心
        mu_u, mu_v = float(np.mean(Up01)), float(np.mean(Vp01))
        if 0.35 <= mu_u <= 0.65 and 0.35 <= mu_v <= 0.65:
            Cb = Up01 - 0.5
            Cr = Vp01 - 0.5
        else:
            Cb = Up01
            Cr = Vp01
        rgb01 = _yuv444_to_rgb01_bt601(Yp01, Cb, Cr, clamp=True)
        _save_color_png(os.path.join(out_dir, f"{stem}_CBM3D_color_preview.png"), rgb01)

    # meta / info
    meta_out = dict(meta)
    meta_out.setdefault("denoise", {})
    meta_out["denoise"].update({
        "method": "cbm3d_numpy_fast",
        "sigma_cap01": float(sigma_cap01),
        "sigma_floor01": float(sigma_floor01),
        "strength": float(strength),
        "chroma_sigma_scale": float(chroma_sigma_scale),
        "colorspace": "YCbCr(YUV444)"
    })
    info = {"backend": "cbm3d_numpy_fast", "sigma_used": sigma_used, "shape": [int(H), int(W)]}

    logger.info("[CBM3D][YUV444] σ(Y/U/V)=%s saved_plots=%s saved_color=%s",
                {k: round(v, 4) for k, v in sigma_used.items()},
                str(bool(save_plots)), str(bool(save_color)))
    return (Y_dn, U_dn, V_dn), meta_out, info
