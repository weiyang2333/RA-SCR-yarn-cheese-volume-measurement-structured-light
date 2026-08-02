import heapq
import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.dual import fft2, ifft2
from numpy.fft import fftfreq
from scipy.signal.windows import hann
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage import measure
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def clear_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在！")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"无法删除 {item_path}，错误：{e}")

def show(Name,img):
    cv2.namedWindow(Name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(Name,img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def process_image(img):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:, :]  # y轴 ,x轴
    # img = guided_filter_denoise(img)
    # img = MSRCR(img)
    show("crop_img",img)
    return img

def process_subfolders(input_root, output_root):
    # 遍历 input_root 下所有子文件夹
    os.makedirs(output_root, exist_ok=True)
    for subfolder in os.listdir(input_root):
        subfolder_path = os.path.join(input_root, subfolder)

        if os.path.isdir(subfolder_path):
            # 创建输出子文件夹
            output_subfolder_path = os.path.join(output_root, subfolder)
            os.makedirs(output_subfolder_path, exist_ok=True)

            # 遍历子文件夹内所有文件
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                # 检查是否为图片文件
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    img = cv2.imread(file_path)

                    # 处理图片
                    processed_img = process_image(img)

                    # 保存到新文件夹
                    save_path = os.path.join(output_subfolder_path, filename)
                    cv2.imwrite(save_path, processed_img)
                    print(f"保存: {save_path}")

#二维解包裹
# 四步相移法获取包裹
#第二个参数 X代表 horizontal
def Si_bu_bao_phase_safe(phase_imgs):



    I0 = cv2.imread(phase_imgs[0], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    I1 = cv2.imread(phase_imgs[1], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    I2 = cv2.imread(phase_imgs[2], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    I3 = cv2.imread(phase_imgs[3], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # 高斯滤波（注意需要将滤波结果赋值回去）
    kernel_size = (3, 3)
    sigma = 0.8
    I0 = cv2.GaussianBlur(I0, kernel_size, sigma)
    I1 = cv2.GaussianBlur(I1, kernel_size, sigma)
    I2 = cv2.GaussianBlur(I2, kernel_size, sigma)
    I3 = cv2.GaussianBlur(I3, kernel_size, sigma)
    numerator = I3 - I1
    denominator = I0 - I2
    # modulation = 0.5 * np.sqrt((I3 - I1) ** 2 + (I0 - I2) ** 2)
    # print("modulate:", modulation)
    # 避免除0问题（虽然 arctan2 本身能处理，但可能引发 warning）
    epsilon = 1e-6
    denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
    phi_wrapped = np.arctan2(numerator, denominator)  # 结果范围 [-π, π]

    # 可选：映射到 [0, 2π]
    phi_wrapped = np.mod(phi_wrapped, 2 * np.pi)

    return phi_wrapped
def Qi_bu_xiangyi_phase_safe(phase_imgs):
    """
    七步相移法相位提取（安全版，支持高斯预滤波）
    phase_imgs: 7 张灰度图路径 [I0,...,I6]
    return: 包裹相位 phi_wrapped (0~2π)
    """
    if len(phase_imgs) != 7:
        raise ValueError("七步相移法需要7张相移图像！")

    # 读入并归一化
    frames = []
    for path in phase_imgs:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        frames.append(img)
    frames = np.stack(frames, axis=0)  # (7, H, W)

    # 七步相移步长：0, 60, 120, 180, 240, 300, 360 度
    deltas = np.linspace(0, 2 * np.pi, 7, endpoint=False)

    # 计算 ∑I*sinδ 和 ∑I*cosδ
    S = np.sum(frames * np.sin(deltas)[:, None, None], axis=0)
    C = np.sum(frames * np.cos(deltas)[:, None, None], axis=0)

    # 避免除零
    epsilon = 1e-6
    C = np.where(np.abs(C) < epsilon, epsilon, C)

    # 包裹相位 [-π, π]
    phi_wrapped = np.arctan2(-S, C)

    # 映射到 [0, 2π]
    phi_wrapped = np.mod(phi_wrapped, 2 * np.pi)

    return phi_wrapped
#以上为最小二乘法提取相位传统方法特例以下为通用最小二乘法提取包裹
def LeastSquares_phase_extract(phase_imgs, deltas, H=1):
    """
    通用最小二乘相位提取（支持谐波拟合抑制）
    输入：
      phase_imgs - N 张相移图像路径
      deltas     - 相移量数组（弧度），长度=N
      H          - 谐波最高阶数（H=1 表示只拟合基频；H=3 可同时拟合三次谐波，抗干扰更强）
    输出：
      phi_wrapped - 包裹相位 (0 ~ 2π)
    """
    # 读入并归一化
    frames = []
    for path in phase_imgs:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.GaussianBlur(img, (3,3), 0.8)  # 可选滤波
        frames.append(img)
    frames = np.stack(frames, axis=0)  # (N, H, W)
    K, H_img, W_img = frames.shape

    # 构建设计矩阵 M = [1, cosδ, sinδ, cos2δ, sin2δ, ...]
    cols = [np.ones(K)]
    for h in range(1, H+1):
        cols.append(np.cos(h * deltas))
        cols.append(np.sin(h * deltas))
    M = np.vstack(cols).T  # (K, 1+2H)

    # 伪逆
    M_pinv = np.linalg.pinv(M)  # (1+2H, K)

    # 每个像素都要解 LS: coeffs = M_pinv @ I
    imgs = frames.reshape(K, -1)            # (K, Npix)
    coeffs = M_pinv @ imgs                  # (1+2H, Npix)  @ 是 矩阵乘法运算符

    # 取基频 cos、sin 系数
    cos1 = coeffs[1, :].reshape(H_img, W_img)
    sin1 = coeffs[2, :].reshape(H_img, W_img)

    phi_wrapped = np.arctan2(sin1, cos1)    # [-π, π]
    phi_wrapped = np.mod(phi_wrapped, 2*np.pi)

    return phi_wrapped

#解包裹算法
def get_neighbors(idx, shape):
    """
    获取二维图像中 (i,j) 点的4邻域像素索引
    """
    i, j = idx
    h, w = shape
    neighbors = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            neighbors.append((ni, nj))
    return neighbors
def quality_guided_unwrap(wrapped_phase):
    """
    质量引导二维相位解包裹算法（基础示例）
    参数：
        wrapped_phase: 包裹相位图像，值域应在 (-π, π] 内
    返回：
        unwrapped_phase: 解包裹后的连续相位图（可能存在全局常数偏移）
    """
    h, w = wrapped_phase.shape

    # 计算质量图：这里简单地采用相邻像素梯度的倒数作为质量指标
    # 计算 x、y 方向的差分（这里使用 np.diff，后面补零使尺寸一致）
    dx = np.diff(wrapped_phase, axis=1)
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')
    dy = np.diff(wrapped_phase, axis=0)
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')
    # 较小的梯度认为质量较高（这里用倒数表示质量，避免除零，加入一个小常数 eps）
    eps = 1e-3
    quality = 1.0 / (np.abs(dx) + np.abs(dy) + eps)

    # 初始化未展开的相位图和标记数组
    unwrapped_phase = np.zeros_like(wrapped_phase)
    visited = np.zeros(wrapped_phase.shape, dtype=bool)

    # 选取质量最高的像素作为种子
    seed = np.unravel_index(np.argmax(quality), quality.shape)
    unwrapped_phase[seed] = wrapped_phase[seed]
    visited[seed] = True

    # 使用优先队列来存储未展开像素，队列的关键字为负质量（因为 heapq 为小根堆）
    heap = []
    for nb in get_neighbors(seed, (h, w)):
        if not visited[nb]:
            heapq.heappush(heap, (-quality[nb], nb, seed))  # (质量, 当前像素, 来源像素)

    while heap:
        # 取出质量最高的候选像素
        neg_q, current, from_pixel = heapq.heappop(heap)
        if visited[current]:
            continue
        # 以邻域中已展开像素作为参考(这里采用来源像素)
        ref = unwrapped_phase[from_pixel]
        wrapped_val = wrapped_phase[current]
        # 计算差值
        delta = wrapped_val - ref
        # 修正相位差，使其落在 (-π, π] 内
        delta_unwrapped = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
        # 得到当前点的展开相位（确保与参考值连续）
        unwrapped_phase[current] = ref + delta_unwrapped
        visited[current] = True

        # 将当前像素的未展开邻域加入队列
        for nb in get_neighbors(current, (h, w)):
            if not visited[nb]:
                heapq.heappush(heap, (-quality[nb], nb, current))

    return unwrapped_phase

#多频相位展开 加复平面平滑以及中值滤波
def complex_smooth(phi_wrapped, ksize=3, iters=1):
    """
    对包裹相位做复平面平滑
    """
    c = np.exp(1j * phi_wrapped)
    for _ in range(iters):
        real = cv2.blur(np.real(c), (ksize, ksize))
        imag = cv2.blur(np.imag(c), (ksize, ksize))
        c = real + 1j * imag
        mag = np.hypot(real, imag) + 1e-12
        c /= mag
    return np.angle(c)


def unwrap_2d_quality_guided(phi_wrapped, mask=None):
    """
    简化版二维质量引导解包裹
    仅用于低频展开结果，保证连续性
    """
    if mask is None:
        mask = np.ones_like(phi_wrapped, dtype=bool)

    # 使用简单的二维 unwrap
    unwrapped = np.unwrap(np.unwrap(phi_wrapped, axis=0), axis=1)
    unwrapped[~mask] = 0  # 对 mask 外区域归零或可插值
    return unwrapped


def robust_multifreq_unwrap(phi_low, phi_mid, phi_high,
                            period_low, period_mid, period_high,
                            mask=None, modulation=None):
    f_low, f_mid, f_high = 1/period_low, 1/period_mid, 1/period_high

    # 平滑
    phi_low = complex_smooth(phi_low)
    phi_mid = complex_smooth(phi_mid)
    phi_high = complex_smooth(phi_high)

    # 改进的低频展开
    phi_low_unw = ls_poisson_unwrap(phi_low, A=modulation, mask=mask)

    # 低频 -> 中频
    k_low_mid = (f_mid/f_low * phi_low_unw - phi_mid) / (2*np.pi)
    k_low_mid = np.round(k_low_mid)
    k_low_mid[~mask] = 0
    k_low_mid = cv2.medianBlur(k_low_mid.astype(np.int16), 3).astype(np.float64)
    phi_mid_unw = phi_mid + k_low_mid * 2 * np.pi

    # 中频 -> 高频
    k_mid_high = (f_high/f_mid * phi_mid_unw - phi_high) / (2*np.pi)
    k_mid_high = np.round(k_mid_high)
    k_mid_high[~mask] = 0
    k_mid_high = cv2.medianBlur(k_mid_high.astype(np.int16), 3).astype(np.float64)
    phi_high_unw = phi_high + k_mid_high * 2 * np.pi

    return  phi_high_unw

#最小二乘解包裹及其配套函数
def compute_A_Imean(images, deltas):
    """
    参数:
        images: list 或 ndarray, 形状 [N, H, W]，N 张相移图像
        deltas: list 或 ndarray, 相移量 (弧度)，长度 N，例如 [0, np.pi/2, np.pi, 3*np.pi/2]

    返回:
        A: 调制度图 (H×W)
        I_mean: 平均亮度图 (H×W)
    """
    frames = []
    for path in images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        frames.append(img)
    frames = np.stack(frames, axis=0)  # (7, H, W)
    images = np.array(frames, dtype=np.float64)  # (N, H, W)
    deltas = np.array(deltas, dtype=np.float64)  # (N,)

    N = len(deltas)
    H, W = images.shape[1], images.shape[2]

    # === Step 1: 计算平均亮度 ===
    I_mean = np.mean(images, axis=0)

    # === Step 2: 计算调制度 A ===
    C = np.zeros((H, W))  # cos 分量
    S = np.zeros((H, W))  # sin 分量
    for k in range(N):
        C += images[k] * np.cos(deltas[k])
        S += images[k] * np.sin(deltas[k])
    A = (2.0 / N) * np.sqrt(C ** 2 + S ** 2)

    return A, I_mean
def huber_weight(r, delta):
    return 1.0 / np.maximum(np.abs(r), delta)
def ls_poisson_unwrap(phi_wrap, A, I_mean=None, mask=None,
                                 Ath_ratio=0.1, Qth=0.2, huber_delta=0.05,
                                 lambda_screened=0.5):
    """
    Screened-Poisson 相位解包裹（增强相位连续性）
    """
    H, W = phi_wrap.shape
    Ath = Ath_ratio * np.nanmax(A)
    if mask is None:
        mask = (A >= Ath)
    mask = mask.astype(bool)

    Q = (A - A.min()) / (A.max() - A.min() + 1e-8)
    Q[~mask] = 0
    mask &= (Q >= Qth)
    mask_use = mask.copy()

    labels = measure.label(mask, connectivity=2)
    n_labels = labels.max()
    phi_unwrap = np.full_like(phi_wrap, np.nan, dtype=np.float64)
    # 存储每个连通域解结果
    region_solutions = {}

    for label in range(1, n_labels+1):
        region = (labels == label)
        if region.sum() < 20:
            continue

        idx_map = -np.ones((H, W), dtype=int)
        idx_map[region] = np.arange(region.sum())
        n_region = region.sum()

        rows, cols, vals = [], [], []
        b = np.zeros(n_region)

        seed = np.unravel_index(np.argmax(Q * region), (H, W))
        seed_id = idx_map[seed]

        for y in range(H):
            for x in range(W):
                if not region[y, x]:
                    continue
                i = idx_map[y, x]
                rows.append(i); cols.append(i); vals.append(lambda_screened * Q[y, x])
                b[i] += lambda_screened * Q[y, x] * phi_wrap[y, x]

                for dy, dx in [(1,0),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if ny>=H or nx>=W: continue
                    if not region[ny,nx]: continue
                    j = idx_map[ny,nx]
                    dphi = np.angle(np.exp(1j*(phi_wrap[y,x] - phi_wrap[ny,nx])))
                    w = Q[y,x]*Q[ny,nx]
                    if I_mean is not None:
                        w *= np.exp(-0.5*abs(I_mean[y,x]-I_mean[ny,nx]))
                    w *= huber_weight(dphi, huber_delta)
                    rows += [i,i,j,j]; cols += [i,j,i,j]; vals += [w,-w,-w,w]
                    b[i] += w*dphi
                    b[j] -= w*dphi

        L = sp.coo_matrix((vals,(rows,cols)), shape=(n_region,n_region)).tocsr()
        L[seed_id,:] = 0
        L[seed_id,seed_id] = 1
        b[seed_id] = phi_wrap[seed]

        phi_region = spla.spsolve(L, b)
        region_solutions[label] = (region, phi_region)

    # === 连通域之间的 2π 对齐 ===
    aligned = set()
    for label, (region, phi_region) in region_solutions.items():
        if not aligned:
            # 第一个区域作为参考
            phi_unwrap[region] = phi_region
            aligned.add(label)
            continue

        # 找与已对齐区域的边界
        ref_mask = np.isin(labels, list(aligned))
        boundary = binary_dilation(ref_mask) & region
        if boundary.sum() == 0:
            # 如果没邻居，先放原始解
            phi_unwrap[region] = phi_region
            aligned.add(label)
            continue

        # 计算相位差并对齐
        delta = np.nanmean(phi_unwrap[boundary]) - np.nanmean(phi_region[region[boundary]])
        shift = np.round(delta / (2*np.pi)) * 2*np.pi
        phi_region += shift

        phi_unwrap[region] = phi_region
        aligned.add(label)

    return phi_unwrap, mask_use
# def least_squares_unwrap(wrapped_phase, mask=None):
#     """
#     Least Squares Phase Unwrapping with optional mask (solves only inside masked region).
#     """
#
#     h, w = wrapped_phase.shape
#
#     if mask is None:
#         mask = np.ones_like(wrapped_phase, dtype=np.float32)
#
#     # 计算包裹相位梯度（使用掩膜）
#     dx = np.angle(np.exp(1j * (np.roll(wrapped_phase, -1, axis=1) - wrapped_phase)))
#     dy = np.angle(np.exp(1j * (np.roll(wrapped_phase, -1, axis=0) - wrapped_phase)))
#
#     # 将不在 mask 区域的梯度清零
#     dx *= mask
#     dy *= mask
#
#     # 散度计算
#     dx_diff = dx - np.roll(dx, 1, axis=1)
#     dy_diff = dy - np.roll(dy, 1, axis=0)
#     div = dx_diff + dy_diff
#
#     # 将不在 mask 区域的散度清零，防止泄漏
#     div *= mask
#
#     # 构建频域解泊松方程
#     fy = fftfreq(h).reshape(-1, 1)
#     fx = fftfreq(w).reshape(1, -1)
#     denom = (2 * np.cos(2 * np.pi * fx) - 2) + (2 * np.cos(2 * np.pi * fy) - 2)
#     denom[0, 0] = 1  # 防止除以 0
#
#     F_div = fft2(div)
#     unwrapped_phase = np.real(ifft2(F_div / denom))
#     unwrapped_phase -= np.mean(unwrapped_phase[mask == 1])  # 只均值中心化 mask 区域
#
#     # 外部区域填充为 0（也可设为 np.nan）
#     unwrapped_phase[mask == 0] = 0
#
#     return unwrapped_phase



#余弦权重滤波
def sinusoidal_weight_filter(phase_map, freq, orientation='horizontal'):
    h, w = phase_map.shape

    # 构造方向性余弦权重掩码
    if orientation == 'horizontal':
        x = np.arange(w)
        weight_1d = 0.5 * (1 + np.cos(2 * np.pi * x / freq))
        weight = np.tile(weight_1d, (h, 1))
    elif orientation == 'vertical':
        y = np.arange(h)
        weight_1d = 0.5 * (1 + np.cos(2 * np.pi * y / freq))
        weight = np.tile(weight_1d[:, np.newaxis], (1, w))
    else:
        raise ValueError("orientation 参数必须是 'horizontal' 或 'vertical'")

    # 构造 2D Hann 窗作为滤波核
    win_size = 5
    win_1d = hann(win_size, sym=True)
    win_2d = np.outer(win_1d, win_1d)

    # 对加权图像与权重掩码分别做卷积
    weighted_phase = phase_map * weight
    filtered = convolve2d(weighted_phase, win_2d, mode='same', boundary='symm')
    weight_sum = convolve2d(weight, win_2d, mode='same', boundary='symm')

    return filtered / (weight_sum + 1e-6)

#质量引导以及残差
def detect_residues(wrapped_phase,radius = 1):
    """
    检测包裹相位图中的残差点
    返回一个矩阵：+1 表示正残差，-1 表示负残差，0 表示无残差
    """
    h, w = wrapped_phase.shape
    residues = np.zeros((h, w), dtype=np.int8)

    # 遍历 2x2 方格
    for y in range(h - 1):
        for x in range(w - 1):
            # 四条边的相位差 (用 exp(iΔφ) 再取 angle 保证在 [-π, π))
            diff1 = np.angle(np.exp(1j * (wrapped_phase[y, x+1] - wrapped_phase[y, x])))
            diff2 = np.angle(np.exp(1j * (wrapped_phase[y+1, x+1] - wrapped_phase[y, x+1])))
            diff3 = np.angle(np.exp(1j * (wrapped_phase[y+1, x] - wrapped_phase[y+1, x+1])))
            diff4 = np.angle(np.exp(1j * (wrapped_phase[y, x] - wrapped_phase[y+1, x])))

            residue = diff1 + diff2 + diff3 + diff4

            # 判断残差类型
            if residue > np.pi:
                residues[y, x] = 1   # 正残差
            elif residue < -np.pi:
                residues[y, x] = -1  # 负残差

    """
        根据残差生成mask
        radius: 把残差点周围 r 邻域也屏蔽
        """
    mask = np.ones_like(residues, dtype=np.uint8)
    y_idx, x_idx = np.nonzero(residues)
    for y, x in zip(y_idx, x_idx):
        y_min, y_max = max(0, y - radius), min(mask.shape[0], y + radius + 1)
        x_min, x_max = max(0, x - radius), min(mask.shape[1], x + radius + 1)
        mask[y_min:y_max, x_min:x_max] = 0
    return mask

def compute_quality_map(wrapped_phase, sigma=1.0):
    """
    计算质量图 (quality map)
    思路：局部梯度一致性越高，质量越好
    """
    # 梯度
    dx = np.angle(np.exp(1j * (np.roll(wrapped_phase, -1, axis=1) - wrapped_phase)))
    dy = np.angle(np.exp(1j * (np.roll(wrapped_phase, -1, axis=0) - wrapped_phase)))

    grad_mag = np.sqrt(dx**2 + dy**2)

    # 平滑 (降低噪声)
    grad_mag_smooth = gaussian_filter(grad_mag, sigma=sigma)

    # 质量 = 梯度小 → 相位更平滑可靠
    quality = 1.0 / (1.0 + grad_mag_smooth)
    mask_quality = (quality > 0.2).astype(np.uint8)

    return mask_quality

#计算残差图即相位连续质量
def compute_residual_map(phi_wrapped: np.ndarray) -> np.ndarray:
    """
    基于相位连续性的残差图计算 (Goldstein 风格的残差检测).
    phi_wrapped: 输入的二维包裹相位 (或解包裹相位) [-pi, pi]
    返回: 残差图 (1表示残差点, 0表示无残差)
    """
    h, w = phi_wrapped.shape
    residual_map = np.zeros((h-1, w-1), dtype=int)

    for i in range(h-1):
        for j in range(w-1):
            # 四个相位点
            phi = [
                phi_wrapped[i, j],
                phi_wrapped[i, j+1],
                phi_wrapped[i+1, j+1],
                phi_wrapped[i+1, j]
            ]

            # 如果有 NaN，跳过这个方格
            if np.any(np.isnan(phi)):
                continue

            # 计算四条边的相位差
            dphi = []
            for k in range(4):
                dp = phi[(k+1) % 4] - phi[k]
                dp = (dp + np.pi) % (2*np.pi) - np.pi
                dphi.append(dp)

            # 判断环路和是否为 ±2π
            total = sum(dphi)
            if np.isnan(total):
                continue
            res = int(round(total / (2*np.pi)))
            if res != 0:
                residual_map[i, j] = res

    return residual_map
#格雷码与移相法结合
def calc_absolute_phase(phi_wrapped, gray_code_index, fringe_period):
    """
    计算绝对相位
    :param phi_wrapped: 包裹相位图 (rad)
    :param gray_code_index: Gray 码解码得到的索引 (int)
    :param fringe_period: 条纹周期 (pixel)
    :return: absolute_phase
    """
    k = gray_code_index.astype(np.float32)
    phi_abs = phi_wrapped + k * 2 * np.pi
    # 也可转换为 p3 坐标
    proj_coord = phi_abs * fringe_period / (2 * np.pi)
    return phi_abs, proj_coord
def ls_poisson_unwrap_v2(
        phi_wrap,A,I_mean,mask=None,Ath_ratio=0.1,Qth=0.1,huber_delta=0.1,
        lambda_screened=1.0,period=None,phi_high=None,alpha_guided=0.3):

    H, W = phi_wrap.shape

    # --- 自适应参数 ---
    if period is not None:
        # 周期越大 → λ越强、HuberΔ越大
        scale = np.clip(period / 16.0, 0.5, 4.0)
        lambda_screened = lambda_screened * (scale ** 1.2)
        huber_delta = huber_delta * (scale ** 0.8)
        Qth = max(0.05, Qth / scale)

    # --- 掩膜生成 ---
    Ath = Ath_ratio * np.nanmax(A)
    if mask is None:
        mask = (A >= Ath)
    mask = mask.astype(bool)
    Q = (A - np.nanmin(A)) / (np.nanmax(A) - np.nanmin(A) + 1e-8)
    Q[~mask] = 0
    mask &= (Q >= Qth)
    mask_use = mask.copy()

    labels = measure.label(mask, connectivity=2)
    n_labels = labels.max()
    phi_unwrap = np.full_like(phi_wrap, np.nan, dtype=np.float64)
    region_solutions = {}

    # --- 遍历每个连通区域 ---
    for label in range(1, n_labels + 1):
        region = (labels == label)
        if region.sum() < 20:
            continue

        idx_map = -np.ones((H, W), dtype=int)
        idx_map[region] = np.arange(region.sum())
        n_region = region.sum()

        rows, cols, vals = [], [], []
        b = np.zeros(n_region)

        seed = np.unravel_index(np.argmax(Q * region), (H, W))
        seed_id = idx_map[seed]

        # 8邻域连接增强连续性
        for y in range(H):
            for x in range(W):
                if not region[y, x]:
                    continue
                i = idx_map[y, x]

                # Screened-Poisson项（平滑约束）
                rows.append(i); cols.append(i)
                vals.append(lambda_screened * Q[y, x])
                b[i] += lambda_screened * Q[y, x] * phi_wrap[y, x]

                # 邻域梯度约束
                for dy, dx in [(1,0), (0,1), (-1,0), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= H or nx < 0 or nx >= W:
                        continue
                    if not region[ny, nx]:
                        continue

                    j = idx_map[ny, nx]
                    dphi = np.angle(np.exp(1j * (phi_wrap[y, x] - phi_wrap[ny, nx])))

                    # 权重组合
                    w = Q[y, x] * Q[ny, nx]
                    if I_mean is not None:
                        w *= np.exp(-0.2 * abs(I_mean[y, x] - I_mean[ny, nx]))
                    w *= huber_weight(dphi, huber_delta)

                    # 构造稀疏矩阵
                    rows += [i, i, j, j]
                    cols += [i, j, i, j]
                    vals += [w, -w, -w, w]
                    b[i] += w * dphi
                    b[j] -= w * dphi

        # --- 组装并求解线性方程组 ---
        L = sp.coo_matrix((vals, (rows, cols)), shape=(n_region, n_region)).tocsr()
        L[seed_id, :] = 0
        L[seed_id, seed_id] = 1
        b[seed_id] = phi_wrap[seed]

        phi_region = spla.spsolve(L, b)
        region_solutions[label] = (region, phi_region)

    # --- 连通域之间的 2π 对齐 ---
    aligned = set()
    for label, (region, phi_region) in region_solutions.items():
        if not aligned:
            phi_unwrap[region] = phi_region
            aligned.add(label)
            continue

        ref_mask = np.isin(labels, list(aligned))
        boundary = binary_dilation(ref_mask) & region
        if boundary.sum() == 0:
            phi_unwrap[region] = phi_region
            aligned.add(label)
            continue

        # 计算相位偏移并对齐
        ref_vals = phi_unwrap[boundary]
        tgt_vals = phi_region[region[boundary]]
        delta = np.nanmean(ref_vals) - np.nanmean(tgt_vals)
        shift = np.round(delta / (2 * np.pi)) * 2 * np.pi
        phi_region += shift
        phi_unwrap[region] = phi_region
        aligned.add(label)

    # --- 高频引导项融合（可选） ---
    if phi_high is not None:
        phi_unwrap = (1 - alpha_guided) * phi_unwrap + alpha_guided * np.angle(np.exp(1j * (phi_high - phi_unwrap)))

    return phi_unwrap, mask_use

#多频融合
# -------------------- 工具函数 --------------------

def robust_mad(x, axis=None, eps=1e-6):
    """Median Absolute Deviation → 鲁棒尺度估计"""
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    return np.squeeze(mad) + eps

def tukey_weights(r, c=4.685):
    """Tukey biweight 权重: |r|>c → 0"""
    w = np.zeros_like(r, dtype=np.float32)
    z = r / float(c)
    m = (np.abs(r) < c)
    w[m] = (1 - z[m]**2)**2
    return w

def normalize_to_geom_units(phases, periods):
    """
    不同频率相位统一到同一几何单位:
        s_k = phi_k * P_k / (2π)
    phases: list[np.ndarray] (K,H,W)，可含 NaN
    periods: list[float]      (K,)
    return: S (K,H,W)
    """
    S = []
    for phi, P in zip(phases, periods):
        S.append(phi * (P / (2*np.pi)))
    return np.stack(S, axis=0)  # (K,H,W)

def build_unified_mask(S, extra_masks=None, keep_largest=True, min_area=4096):
    """
    统一质量掩膜：各频非 NaN ∩ 额外掩膜 ∩ 最大连通域
    """
    valid = np.isfinite(S).all(axis=0)
    if extra_masks:
        for m in extra_masks:
            if m is None:
                continue
            valid &= (m.astype(bool))
    if not valid.any():
        return valid

    if keep_largest:
        # 只保留最大连通域，避免零散噪声干扰
        u8 = valid.astype(np.uint8)
        num, labels = cv2.connectedComponents(u8)
        if num > 1:
            areas = [(labels == i).sum() for i in range(1, num)]
            i_max = int(np.argmax(areas)) + 1
            valid2 = (labels == i_max)
            if valid2.sum() >= min_area:
                valid = valid2
    return valid

def repair_short_2pi_jumps(phi, mask, axis=1, jump_thresh=np.pi, max_run=8):
    """
    短段 ±2π 连续性修复：沿主方向扫描，仅在较短的断裂处补偿，避免大范围误修。
    phi: (H,W) 相位(弧度), 可含 NaN
    mask: (H,W) 有效掩膜
    axis: 1→沿 x（行内），0→沿 y（列内）
    """
    out = phi.copy()
    H, W = phi.shape
    if axis == 1:
        for y in range(H):
            row_mask = mask[y, :]
            if row_mask.sum() < 2:
                continue
            idx = np.where(row_mask)[0]
            vals = out[y, idx].astype(np.float64)
            # 逐点累加修正
            for k in range(1, len(vals)):
                d = vals[k] - vals[k-1]
                if np.abs(d) > jump_thresh and (k < max_run or (len(vals)-k) < max_run):
                    # 就近的±2π校正
                    vals[k:] -= np.round(d / (2*np.pi)) * (2*np.pi)
            out[y, idx] = vals
    else:
        for x in range(W):
            col_mask = mask[:, x]
            if col_mask.sum() < 2:
                continue
            idx = np.where(col_mask)[0]
            vals = out[idx, x].astype(np.float64)
            for k in range(1, len(vals)):
                d = vals[k] - vals[k-1]
                if np.abs(d) > jump_thresh and (k < max_run or (len(vals)-k) < max_run):
                    vals[k:] -= np.round(d / (2*np.pi)) * (2*np.pi)
            out[idx, x] = vals
    return out

# -------------------- 核心：稳健多频外差融合 --------------------

def robust_multifreq_heterodyne(
    phases,
    periods,
    base_weights=None,
    extra_masks=None,
    ref_mode="highest",
    pref=None,
    do_repair=True,
    repair_axis=1,
    irls_iters=3,
    tukey_c=4.685,
    preserve_wrap=False  # 新增：是否保留多周期相位信息
):
    """
    返回:
        phi_fused  : (H,W) 融合相位(弧度, 对应 pref 周期, 已统一到 [0,2π) )
        valid_mask : (H,W)
        weights    : (K,H,W)
        residual   : (K,H,W)
    """
    import numpy as np
    K = len(phases)
    H, W = phases[0].shape

    # 1) 转几何单位
    S = np.stack([phases[k] * periods[k] / (2*np.pi) for k in range(K)], axis=0)
    valid = np.ones((H, W), dtype=bool)
    if extra_masks is not None:
        for m in extra_masks:
            valid &= m

    # 2) 选择参考
    if isinstance(ref_mode, int):
        ref_idx = ref_mode
    elif ref_mode == "highest":
        ref_idx = int(np.argmin(periods))
    elif ref_mode == "middle":
        ref_idx = int(np.argsort(periods)[len(periods)//2])
    else:
        ref_idx = int(np.argmin(periods))
    Sref = S[ref_idx]

    # 3) 残差与权重
    resid = np.abs(S - Sref[None, :, :])
    scale = np.nanmedian(resid, axis=(1,2))[ref_idx] + 1e-9
    W_cons = np.exp(-0.5*(resid/scale)**2)
    W_base = np.ones_like(W_cons)
    W_tot = W_cons * W_base
    W_tot *= valid[None, :, :].astype(np.float32)

    # 4) IRLS 融合
    denom = np.sum(W_tot, axis=0) + 1e-9
    s_star = np.sum(W_tot * S, axis=0) / denom

    # 5) 回到相位
    if pref is None:
        pref = periods[ref_idx]
    phi_star = s_star * (2*np.pi / pref)

    # 6) ±2π修复（如需）
    if do_repair:
        phi_star = repair_short_2pi_jumps(phi_star, valid, axis=repair_axis)

    # 7) 归一化到 [0,2π)
    if not preserve_wrap:
        phi_star = np.mod(phi_star, 2*np.pi)

    # 附带输出
    return phi_star, valid, W_tot, resid
