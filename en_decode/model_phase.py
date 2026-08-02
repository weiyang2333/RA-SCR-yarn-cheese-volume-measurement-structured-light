import glob
import re
import sys
import time

from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from skimage import morphology
from skimage import measure
import pyamg
import numpy as np
import cv2
import os
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.ndimage import distance_transform_edt

def show(Name,img):
    cv2.namedWindow(Name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(Name,img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
def unwrap_multifreq(phi_low, phi_mid, phi_high, lambda_low, lambda_mid, lambda_high,
                     do_spatial_unwrap=True, smooth_sigma=1.0):
    """
    稳健的三频逐级展开（low -> mid -> high）
    - 先对低频做空间解包裹（行/列 np.unwrap），再做可选高斯平滑
    - 使用 low_unwrapped 计算 mid 的 k，再用 mid_unwrapped 计算 high 的 k
    返回: phi_mid_unwrap, phi_high_unwrap （连续相位，弧度）
    """
    # 保证包裹在 [-pi, pi]
    phi_low = np.angle(np.exp(1j * phi_low))
    phi_mid = np.angle(np.exp(1j * phi_mid))
    phi_high = np.angle(np.exp(1j * phi_high))

    # 1) spatial unwrap low (recommended)
    if do_spatial_unwrap:
        # 在两个方向上做 unwrap（行、列）
        low_spatial = np.unwrap(np.unwrap(phi_low, axis=1), axis=0)
        # 小幅平滑以抑制噪声对 k 估计的影响
        if smooth_sigma is not None and smooth_sigma > 0:
            low_spatial = gaussian_filter(low_spatial, sigma=smooth_sigma, mode='reflect')
    else:
        low_spatial = phi_low.copy()

    # 2) low -> mid
    k_mid = np.round(((lambda_low / lambda_mid) * low_spatial - phi_mid) / (2.0 * np.pi))
    phi_mid_unwrap = phi_mid + 2.0 * np.pi * k_mid

    # Optional: spatially smooth mid_unwrap a bit to stabilize next step
    phi_mid_unwrap_s = gaussian_filter(phi_mid_unwrap, sigma=0.5, mode='reflect')

    # 3) mid -> high
    k_high = np.round(((lambda_mid / lambda_high) * phi_mid_unwrap_s - phi_high) / (2.0 * np.pi))
    phi_high_unwrap = phi_high + 2.0 * np.pi * k_high

    return phi_mid_unwrap, phi_high_unwrap
def unwrap_with_cue(phi_high_wrapped, phi_cue_abs, f_high):

    mask_valid = np.isfinite(phi_high_wrapped) & np.isfinite(phi_cue_abs)
    phi_expected = phi_cue_abs * f_high
    P = np.rint((phi_expected - phi_high_wrapped) / (2*np.pi))

    P[~mask_valid] = np.nan
    phi_high_unwrapped = phi_high_wrapped + P * (2*np.pi)
    # phi_absolute = phi_high_unwrapped / f_high
    # 你的实际相位范围（从数据得出）
    phi_min = np.nanmin(phi_high_unwrapped)
    phi_max = np.nanmax(phi_high_unwrapped)
    #平移，使最小值变成 0
    phi_shift = phi_high_unwrapped - phi_min

    return phi_shift
def unwrap(phi37_wrapped, phi10_wrapped, f_high=37, f_mid=10):
    """
    使用 f10 包裹相位展开 f37 包裹相位（两个输入均为 [-π, π] 区间内的包裹相位）
    返回：phi37_abs_shift —— f37 的连续绝对相位（最小值平移到 0）
    """
    # 1. 有效 mask
    mask_valid = np.isfinite(phi37_wrapped) & np.isfinite(phi10_wrapped)
    phi_expected = (f_high / f_mid) * phi10_wrapped     # 37/10 * φ10
    P = np.rint((phi_expected - phi37_wrapped) / (2 * np.pi))
    P[~mask_valid] = np.nan

    phi37_unwrapped = phi37_wrapped + P * (2 * np.pi)
    phi37_unwrapped_n = phi37_unwrapped / f_high
    phi_min = np.nanmin(phi37_unwrapped_n)
    phi37_abs_shift = phi37_unwrapped_n - phi_min

    return phi37_abs_shift

def hierarchical_unwrap(phi_low, phi_mid, phi_high, f_low, f_mid, f_high):
    """
    层级相位展开（Hierarchical Phase Unwrapping, TPU）
    输入：
        phi_low  : 最低频包裹相位 (0~2π)
        phi_mid  : 中频包裹相位 (0~2π)
        phi_high : 高频包裹相位 (0~2π)
        f_low, f_mid, f_high : 对应频率（整数）
    输出：
        phi_low_abs, phi_mid_abs, phi_high_abs : 绝对相位
    """

    # --- 转为 [-π, π) ---
    phi_low  = np.angle(np.exp(1j * phi_low))
    phi_mid  = np.angle(np.exp(1j * phi_mid))
    phi_high = np.angle(np.exp(1j * phi_high))

    mask = np.isfinite(phi_low) & np.isfinite(phi_mid) & np.isfinite(phi_high)

    # =======================================================
    # 1. φ_low 是基准（必须不跨 2π）
    # =======================================================
    phi_low_abs = phi_low.copy()

    # =======================================================
    # 2. 用 φ_low_abs 展开 φ_mid
    # =======================================================
    # 预测中频相位:
    phi_mid_pred = phi_low_abs * (f_mid / f_low)

    # 求整数 k_mid:
    k_mid = np.rint((phi_mid_pred - phi_mid) / (2 * np.pi))

    phi_mid_abs = phi_mid + 2 * np.pi * k_mid
    phi_mid_abs[~mask] = np.nan

    # =======================================================
    # 3. 用 φ_mid_abs 展开 φ_high
    # =======================================================
    phi_high_pred = phi_mid_abs * (f_high / f_mid)
    k_high = np.rint((phi_high_pred - phi_high) / (2 * np.pi))

    phi_high_abs = phi_high + 2 * np.pi * k_high
    phi_high_abs[~mask] = np.nan

    return phi_low_abs, phi_mid_abs, phi_high_abs

def Het_unwrap(philow, phimid, phihigh, f1, f2, f3):
    # --- 统一为主值相位 [-π, π) ---
    philow = np.angle(np.exp(1j * philow))
    phimid = np.angle(np.exp(1j * phimid))
    phihigh = np.angle(np.exp(1j * phihigh))

    mask = np.isfinite(philow) & np.isfinite(phimid) & np.isfinite(phihigh)
    mask_uint8 = (mask.astype(np.uint8) * 255)

    feq = abs(f3 - f2)  # =1
    # 差频包裹相位 φ_eq = wrap(φ37 − φ36)
    phi_eq = np.angle(np.exp(1j * (phihigh - phimid)))
    phi_eq[phi_eq < 0] += 2 * np.pi  # 很重要 把负相位扭转为正值避免后续丢失
    print(np.nanmin(phi_eq), np.nanmax(phi_eq))
    # show("phi_eq", phi_eq)
    # phi_eq 范围 [-π, π)，但周期极长 → 可视为“粗相位”
    k36 = np.rint(((f2 / feq) * phi_eq - phimid) / (2 * np.pi))
    phi36_abs = phimid + k36 * (2 * np.pi)
    phi36_abs[~mask] = np.nan

    k37 = np.rint(((f3 / f2) * phi36_abs - phihigh) / (2 * np.pi))
    phi37_abs = phihigh + k37 * (2 * np.pi)
    phi37_abs[~mask] = np.nan
    # phase_pit(phi_eq,"phi_eq", phi36_abs,"phi36_abs",phi37_abs,"phi37_abs")

    k10 = np.rint(((f1 / feq) * phi_eq - philow) / (2 * np.pi))
    phi10_abs = philow + k10 * (2 * np.pi)
    phi10_abs[~mask] = np.nan


    return phi10_abs/f1, phi36_abs/f2, phi37_abs/f3


#加权修正
def normalize(x,eps = 1e-6):
    x = x.astype(np.float64)
    mask = np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float64)
    if np.any(mask):
        xmin = np.nanpercentile(x[mask], 2)
        xmax = np.nanpercentile(x[mask], 98)
        if xmax - xmin < eps:
            out[mask] = 0.0
        else:
            out[mask] = np.clip((x[mask] - xmin) / (xmax - xmin), 0, 1)
    return out

#前置工具


def circular_mean(arr):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return np.angle(np.nanmean(np.exp(1j * arr)))
def clean_noise(phi_abs):
    mask_object = (phi_abs > 0)  # 或 phi_abs > threshold
    mask_object = morphology.remove_small_objects(mask_object, min_size=400)
    phi_abs_clean_x = phi_abs.copy()
    phi_abs_clean_x[~mask_object] = np.nan
    return phi_abs_clean_x
#最小二乘法
def compute_A_Imean(frames, deltas):

    # frames = []
    # for path in images:
    #     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    #     frames.append(img)
    eps = 1e-6
    frames = np.stack(frames, axis=0)  # (7, H, W)
    images = np.array(frames, dtype=np.float64)  # (N, H, W)
    deltas = np.array(deltas, dtype=np.float64)  # (N,)

    N = len(deltas)
    H, W = images.shape[1], images.shape[2]
    I_mean = np.mean(images, axis=0)

    C = np.zeros((H, W))  # cos 分量
    S = np.zeros((H, W))  # sin 分量
    for k in range(N):
        C += images[k] * np.cos(deltas[k])
        S += images[k] * np.sin(deltas[k])
    A = (2.0 / N) * np.sqrt(C ** 2 + S ** 2)
    Q_m = A / (I_mean + eps)
    return A, I_mean,Q_m

def contrast_quality(frames, win=5, eps=1e-6):
    """
    Qc: 条纹对比度质量
    根据移相图逐像素统计局部对比度
    """
    imgs = np.stack(frames, axis=0).astype(np.float64)   # (N,H,W)

    local_q = []
    kernel = (win, win)

    for k in range(imgs.shape[0]):
        img = imgs[k]
        local_max = cv2.dilate(img, np.ones(kernel, np.uint8))
        local_min = cv2.erode(img, np.ones(kernel, np.uint8))
        c = (local_max - local_min) / (local_max + local_min + eps)
        local_q.append(c)

    q = np.mean(np.stack(local_q, axis=0), axis=0)
    return normalize(q)
def phase_residual_quality(frames, phi_wrapped, deltas, eps=1e-6):
    """
    Qr: 相位残差质量
    用移相模型重建每一步理论图像，再和真实图像比较
    """
    imgs = np.stack(frames, axis=0).astype(np.float64)   # (N,H,W)
    deltas = np.array(deltas, dtype=np.float64)

    A, I_mean,Qm = compute_A_Imean(frames, deltas)

    # phi_wrapped 统一到 [-pi, pi]
    phi = np.angle(np.exp(1j * phi_wrapped))

    residual = np.zeros_like(phi, dtype=np.float64)
    N = len(deltas)

    for k in range(N):
        I_hat = I_mean + A * np.cos(phi + deltas[k])
        residual += (imgs[k] - I_hat) ** 2

    residual /= N

    # 残差越小越好，所以取反
    q = 1.0 - normalize(residual)
    q[~np.isfinite(phi_wrapped)] = 0
    return q
def compute_phase_quality(frames, phi_wrapped, deltas,
                          wm=0.5, wc=0.2, wr=0.3):
    """
    综合质量图 Q = wm*Qm + wc*Qc + wr*Qr
    """
    A, I_mean,Qm = compute_A_Imean(frames, deltas)
    Qc = contrast_quality(frames)
    Qr = phase_residual_quality(frames, phi_wrapped, deltas)

    Q = wm * Qm + wc * Qc + wr * Qr
    Q = normalize(Q)
    Q[~np.isfinite(phi_wrapped)] = 0

    return Q, Qm, Qc, Qr
def local_phase_repai(phi, Q, Q_th=0.45, min_neighbors=4, max_iter=5):
    """
    快速局部受限传播修复（加权平均版）

    参数:
        phi : 2D 相位图
        Q   : 2D 质量图，与 phi 同尺寸
        Q_th : 高可信阈值
        min_neighbors : 至少需要多少个邻域有效点才允许修复
        max_iter : 最大迭代次数

    返回:
        phi_new : 修复后的相位图
        state   : 状态图
                  0 = 低质量/未修复
                  1 = 修复后的次级可信点（最多再参与一次传播）
                  2 = 原始高可信点
    """
    phi_new = phi.copy()
    H, W = phi.shape

    # 0 = 低质量/未修复
    # 1 = 修复后的次级可信点（最多再参与一次传播）
    # 2 = 原始高可信点
    state = np.zeros((H, W), dtype=np.uint8)
    state[(Q > Q_th) & np.isfinite(phi)] = 2

    # 次级可信点还能参与传播的次数
    secondary_use_left = np.zeros((H, W), dtype=np.uint8)

    # 8邻域及距离权重
    neighbors = [
        (-1, -1, 1.0 / np.sqrt(2)),
        (-1,  0, 1.0),
        (-1,  1, 1.0 / np.sqrt(2)),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
        ( 1, -1, 1.0 / np.sqrt(2)),
        ( 1,  0, 1.0),
        ( 1,  1, 1.0 / np.sqrt(2)),
    ]

    for _ in range(max_iter):
        changed = 0

        # 冻结上一轮结果，避免当轮新修点立刻继续传播
        phi_old = phi_new.copy()
        state_old = state.copy()
        use_old = secondary_use_left.copy()

        secondary_used_this_round = np.zeros((H, W), dtype=bool)

        # 只遍历坏点
        ys, xs = np.where(state_old == 0)

        for y, x in zip(ys, xs):
            if y == 0 or y == H - 1 or x == 0 or x == W - 1:
                continue

            weighted_sum = 0.0
            weight_sum = 0.0
            valid_count = 0

            for dy, dx, w_dist in neighbors:
                ny = y + dy
                nx = x + dx

                val = phi_old[ny, nx]
                if not np.isfinite(val):
                    continue

                s = state_old[ny, nx]

                # 原始高可信点始终可用
                if s == 2:
                    pass
                # 次级可信点只能再参与一次传播
                elif s == 1 and use_old[ny, nx] > 0:
                    secondary_used_this_round[ny, nx] = True
                else:
                    continue

                # 最终权重 = 邻点质量 * 距离权重
                w = float(Q[ny, nx]) * w_dist
                if w <= 0:
                    continue

                weighted_sum += w * val
                weight_sum += w
                valid_count += 1

            if valid_count < min_neighbors or weight_sum <= 1e-12:
                continue

            phi_fit = weighted_sum / weight_sum

            if np.isfinite(phi_fit):
                phi_new[y, x] = phi_fit
                state[y, x] = 1
                secondary_use_left[y, x] = 1
                changed += 1

        # 扣减本轮被使用过的次级可信点的传播次数
        used_mask = (state_old == 1) & secondary_used_this_round & (use_old > 0)
        secondary_use_left[used_mask] -= 1
        secondary_use_left[secondary_use_left < 0] = 0

        if changed == 0:
            break

    return phi_new, state

#展示
def phase_pit(p1,p1name,p2,p2name,p3,p3name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 第一张
    axes[0].imshow(p1, cmap='hsv')
    axes[0].set_title(str(p1name))
    axes[0].set_axis_off()
    fig.colorbar(axes[0].images[0], ax=axes[0])

    # 第二张
    axes[1].imshow(p2, cmap='hsv')
    axes[1].set_title(str(p2name))
    axes[1].set_axis_off()
    fig.colorbar(axes[1].images[0], ax=axes[1])

    # 第三张
    axes[2].imshow(p3, cmap='hsv')
    axes[2].set_title(str(p3name))
    axes[2].set_axis_off()
    fig.colorbar(axes[2].images[0], ax=axes[2])

    plt.tight_layout()
    plt.show()
def show_phase(title,phi,cmap="hsv"):
    plt.figure(figsize=(6,5))
    plt.imshow(phi, cmap=cmap)
    plt.colorbar()
    # plt.title(title)
    plt.tight_layout()
    plt.show()
def show_histogram(name,data, bins=200):
    """
    查看灰度/相位数据直方图（支持 .npy 或图像文件）
    自动打印 min, max, mean, std 并绘制分布曲线。

    参数：
        path : str  文件路径（.npy, .png, .jpg 等）
        bins : int  直方图分箱数量
    """
    # --- 加载数据 ---
    if data.ndim == 3:  # 如果是彩色图，转灰度
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    # --- 转 float 并去掉无效值 ---
    data = data.astype(np.float64)
    data = data[np.isfinite(data)]

    # --- 打印统计信息 ---
    # print(f"[INFO] File: {path}")
    # print(f"       shape: {data.shape}")
    print(f"       min={np.nanmin(data):.6f}, max={np.nanmax(data):.6f}")
    print(f"       mean={np.nanmean(data):.6f}, std={np.nanstd(data):.6f}")

    # --- 绘制直方图 ---
    plt.figure(figsize=(8, 4))
    plt.hist(data.ravel(), bins=bins, color='steelblue', edgecolor='black', alpha=0.75)
    plt.title(f"Histogram of {os.path.basename(str(name))}")
    plt.xlabel("Value")
    plt.ylabel("NUM")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

