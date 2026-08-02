import glob
import math
import re
import sys
sys.path.append(r"D:\PythonDoc\Structure_Light\opencv")
from model_photo import *
import cv2
import numpy as np
import os
import subprocess
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

def get_sorted_gray_paths(graycode_dir):
    def get_num(path):
        filename = os.path.basename(path)
        return int(filename.split("_")[-1].split(".")[0])

    gray_paths = glob.glob(os.path.join(graycode_dir, "*gray*.png"))
    gray_paths.sort(key=get_num)
    return gray_paths
def decode_graycode_full(gray_files,black_img,white_img, projector_size):

    pattern_images = []
    # === H方向 ===
    for bit in range(10):
        pattern_images.append(gray_files[28 + bit])  # 正
        pattern_images.append(gray_files[18 + bit])  # 负
    # === V方向 ===
    for bit in range(9):
        pattern_images.append(gray_files[9 + bit])  # 正
        pattern_images.append(gray_files[0 + bit])  # 负

    #注意：OpenCV 要求先 White 再 Black
    pattern_images.append(white_img)
    pattern_images.append(black_img)
    for idx, img in enumerate(pattern_images):
        assert img is not None, f"pattern[{idx}] is None"
        assert img.ndim == 2, f"pattern[{idx}] 不是灰度图"
        assert img.dtype == np.uint8, f"pattern[{idx}] dtype={img.dtype}"
        assert img.shape == pattern_images[0].shape, f"pattern[{idx}] shape 不一致"
    print("✅ 所有图像检查通过")

    h, w = pattern_images[0].shape
    print(f"输入图像尺寸: {w}x{h}")
    graycode = cv2.structured_light_GrayCodePattern.create(854,480)
    graycode.setWhiteThreshold(8)
    graycode.setBlackThreshold(0)
    H, W = pattern_images[0].shape
    proj_coords_h = np.full((H, W), np.nan, np.float32)
    proj_coords_v = np.full((H, W), np.nan, np.float32)
    mask_valid = np.zeros((H, W), np.uint8)
    total = H * W
    print(f"[INFO] 开始全图 Gray 解码: {H}x{W}, 共 {total:,} 像素")
    # 批量遍历
    for y in range(H):
        for x in range(W):
            err, proj_pix = graycode.getProjPixel(pattern_images, x, y)
            if not err:
                u, v = proj_pix
                proj_coords_h[y, x] = u
                proj_coords_v[y, x] = v
                mask_valid[y, x] = 255
        if y % 50 == 0:
            print(f"  → 解码进度 {y}/{H}")
    # 限制在投影仪分辨率范围内
    proj_coords_h = np.clip(proj_coords_h, 0, projector_size[0]-1)
    proj_coords_v = np.clip(proj_coords_v, 0, projector_size[1]-1)
    #提取有用参数

    return proj_coords_h, proj_coords_v, mask_valid
def decode_phase():
    script_path = r"D:\PythonDoc\Structure_Light\en_decode\decode phase.py"
    u_path = r"D:\PythonDoc\Structure_Light\en_decode\mid\u_calib_pred.npy"
    v_path = r"D:\PythonDoc\Structure_Light\en_decode\mid\v_calib_pred.npy"
    subprocess.run([sys.executable, script_path], check=True)
    u_pred = np.load(u_path).astype(np.float32)
    v_pred = np.load(v_path).astype(np.float32)
    return u_pred, v_pred

def wrap_2pi(phi):
    return np.mod(phi, 2 * np.pi)
def circular_median_offset(meas, ref):
    """
    估计 meas 相对于 ref 的相位偏移
    """
    diff = np.angle(np.exp(1j * (meas - ref)))
    return np.angle(np.nanmean(np.exp(1j * diff)))
def fuse_gray_phase_keep_gray(gray_u, gray_v, phase_u, phase_v, max_diff=5.0):
    gray_u = gray_u.astype(np.float32)
    gray_v = gray_v.astype(np.float32)
    phase_u = phase_u.astype(np.float32)
    phase_v = phase_v.astype(np.float32)

    out_u = gray_u.copy()
    out_v = gray_v.copy()

    valid_gray = (
        np.isfinite(gray_u) &
        np.isfinite(gray_v) &
        (gray_u >= 0) & (gray_u < 854) &
        (gray_v >= 0) & (gray_v < 480)
    )

    valid_phase = (
        np.isfinite(phase_u) &
        np.isfinite(phase_v) &
        (phase_u >= 0) & (phase_u < 854) &
        (phase_v >= 0) & (phase_v < 480)
    )

    diff_u = np.abs(phase_u - gray_u)
    diff_v = np.abs(phase_v - gray_v)

    replace = valid_gray & valid_phase & (diff_u < max_diff) & (diff_v < max_diff)

    # 只在相位可靠的地方，用相位替换 Gray
    out_u[replace] = phase_u[replace]
    out_v[replace] = phase_v[replace]

    # Gray 无效的地方仍然设为 NaN
    out_u[~valid_gray] = np.nan
    out_v[~valid_gray] = np.nan

    print("[fuse] gray valid:", np.count_nonzero(valid_gray))
    print("[fuse] phase replace:", np.count_nonzero(replace))
    print("[fuse] replace ratio:", np.count_nonzero(replace) / np.count_nonzero(valid_gray))

    return out_u, out_v
def gray_guided_phase_refine(gray_coord, phi_wrapped, width, freq, max_diff=5.0):
    """
    Gray 引导高频包裹相位展开：
    gray_coord: Gray 解码得到的投影仪坐标，例如 gray_u
    phi_wrapped: 高频包裹相位，例如 phi_wrapped_high_x
    width: 对应方向投影仪尺寸，u方向854，v方向480
    freq: 高频频率，例如37
    """

    gray_coord = gray_coord.astype(np.float32)
    phi_wrapped = phi_wrapped.astype(np.float32)

    valid = np.isfinite(gray_coord) & np.isfinite(phi_wrapped)

    # Gray 坐标对应的理论高频相位
    phi_ref = wrap_2pi(2 * np.pi * freq * gray_coord / width)

    # 判断相位方向是否反了
    offset_pos = circular_median_offset(phi_wrapped[valid], phi_ref[valid])
    phi_pos = wrap_2pi(phi_wrapped - offset_pos)
    err_pos = np.nanmedian(np.abs(np.angle(np.exp(1j * (phi_pos[valid] - phi_ref[valid])))))

    offset_neg = circular_median_offset(-phi_wrapped[valid], phi_ref[valid])
    phi_neg = wrap_2pi(-phi_wrapped - offset_neg)
    err_neg = np.nanmedian(np.abs(np.angle(np.exp(1j * (phi_neg[valid] - phi_ref[valid])))))

    if err_neg < err_pos:
        phi_corr = phi_neg
        print("[gray-guided] use flipped phase")
    else:
        phi_corr = phi_pos
        print("[gray-guided] use normal phase")

    # 用 Gray 坐标确定周期号
    phi_expected_unwrapped = 2 * np.pi * freq * gray_coord / width
    k = np.round((phi_expected_unwrapped - phi_corr) / (2 * np.pi))

    phi_unwrapped = phi_corr + 2 * np.pi * k

    # 转成投影仪连续坐标
    refined = phi_unwrapped * width / (2 * np.pi * freq)

    # 再用 Gray 做最后一致性筛选
    diff = np.abs(refined - gray_coord)

    out = np.full_like(gray_coord, np.nan, dtype=np.float32)
    out[valid & (diff < max_diff)] = refined[valid & (diff < max_diff)]

    print("[gray-guided] valid:", np.count_nonzero(np.isfinite(out)))
    print("[gray-guided] ratio:", np.count_nonzero(np.isfinite(out)) / out.size)
    print("[gray-guided] diff median:", np.nanmedian(diff[valid]))

    return out

def calibrate_camera(img_dir, square_size,board_size):
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints,px_dists = [], [],[]
    # 角点亚像素精炼条件

    images = glob.glob(os.path.join(img_dir, "*.png"))
    img_list = []
    for fname in images:
        img = cv2.imread(fname)
        img_list.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not ret:
            continue
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)
            # 计算相邻角点间的平均像素距离（横向 + 纵向）
        corners2 = corners2.reshape(board_size[1], board_size[0], 2)
        dists_h = np.linalg.norm(np.diff(corners2, axis=1), axis=2)  # 横向距离
        dists_v = np.linalg.norm(np.diff(corners2, axis=0), axis=2)  # 纵向距离

        px_mean = np.mean(np.concatenate([dists_h.flatten(), dists_v.flatten()]))
        px_dists.append(px_mean)

    px_per_square = np.mean(px_dists)
    mm_per_px = square_size / px_per_square  #具体将物理世界坐标转为像素坐标目前没有用到

    flags = (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 |
             cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1],
        None, None, flags=flags)

    # for i in range(len(objpoints)):
    #     objpoints[i] = objpoints[i] / mm_per_px  # 单位从 mm → px
    print("相机标定质量平均重投影误差：",ret)
    print(f"[INFO] 平均每个方格像素宽度: {px_per_square:.2f} px")
    # print(f"[INFO] mm_per_px = {mm_per_px:.4f} mm/px")
    # print(f"[INFO] px_per_mm = {1 / mm_per_px:.2f} px/mm")
    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints,img_list,mm_per_px
def fix_chessboard_orientation(corner_points, board_width, board_height):
    """
    自动修正棋盘角点方向，使得角点排列方向统一（左上→右下）
    corner_points: Nx2 或 Nx1x2
    """
    pts = corner_points.reshape(-1, 2).astype(np.float32)

    # 计算四个角的像素坐标：第一个点、最后一个点、中间点
    p00 = pts[0]                  # (0,0)
    pW0 = pts[board_width - 1]    # (W-1,0)
    p0H = pts[(board_height - 1) * board_width]          # (0,H-1)
    pWH = pts[-1]                 # (W-1,H-1)

    # 判断棋盘整体走向
    # 使用 p00→pW0 的向量判断 x 方向
    # 使用 p00→p0H 的向量判断 y 方向
    vec_x = pW0 - p00
    vec_y = p0H - p00

    flip_x = vec_x[0] < 0   # x 方向反向：右边比左边还小 → 翻转
    flip_y = vec_y[1] < 0   # y 方向反向：下边比上边小 → 翻转

    pts = pts.reshape(board_height, board_width, 2)

    if flip_x:
        pts = np.flip(pts, axis=1)

    if flip_y:
        pts = np.flip(pts, axis=0)

    return pts.reshape(-1, 2)
def calib_points(objpoints_world_list,imgpoints_cam_list,
        proj_coords_h,proj_coords_v,number,patch_init=5):
    projector_size = (854, 480)
    proj_w, proj_h = projector_size
    H_cam, W_cam = proj_coords_h.shape
    proj_coords_h = np.where(proj_coords_h < 0, np.nan, proj_coords_h)
    proj_coords_v = np.where(proj_coords_v < 0, np.nan, proj_coords_v)
    map_h = np.clip(proj_coords_h, 0, proj_w - 1).astype(np.float32)
    map_v = np.clip(proj_coords_v, 0, proj_h - 1).astype(np.float32)

    MIN_NEIGH_POINTS = 16
    MAX_PATCH = 12
    RANSAC_REPROJ = 3.0
    MIN_INLIER_RATIO = 0.55
    CONSISTENCY_ERR = 2.0

    proj_objpoints = []
    proj_imgpoints = []
    stereo_imgpoints_cam = []

    print(f"[INFO] 初始 patch = {patch_init}")
    for img_id, (objp_world, imgp_cam) in enumerate(zip(objpoints_world_list, imgpoints_cam_list)):
        objp_world = objp_world.reshape(-1, 3)
        imgp_cam = imgp_cam.reshape(-1, 2)
        imgp_cam = fix_chessboard_orientation(imgp_cam, 11, 8)
        proj_pts = []
        obj_pts = []
        cam_pts = []
        for idx, (cx, cy) in enumerate(imgp_cam):
            cx_f, cy_f = float(cx), float(cy)
            local_patch = patch_init
            src_points = []
            dst_points = []
            tmp_count = 0
            for dy in range(-patch_init, patch_init + 1):
                for dx in range(-patch_init, patch_init + 1):
                    x = int(cx + dx)
                    y = int(cy + dy)
                    if 0 <= x < W_cam and 0 <= y < H_cam:
                        u = map_h[y, x]
                        v = map_v[y, x]
                        if np.isfinite(u) and np.isfinite(v):
                            tmp_count += 1
            while local_patch <= MAX_PATCH:
                src_points.clear()
                dst_points.clear()
                x0 = int(math.floor(cx_f - local_patch))
                x1 = int(math.ceil(cx_f + local_patch))
                y0 = int(math.floor(cy_f - local_patch))
                y1 = int(math.ceil(cy_f + local_patch))
                for y in range(y0, y1 + 1):
                    if not (0 <= y < H_cam):
                        continue
                    for x in range(x0, x1 + 1):
                        if not (0 <= x < W_cam):
                            continue
                        u = map_h[y, x]
                        v = map_v[y, x]
                        if np.isfinite(u) and np.isfinite(v):
                            src_points.append([float(x), float(y)])
                            dst_points.append([float(u), float(v)])
                if len(src_points) >= MIN_NEIGH_POINTS:
                    break
                local_patch += 1
            # 不够
            if len(src_points) < 4:
                continue
            src_points = np.array(src_points, dtype=np.float32)
            dst_points = np.array(dst_points, dtype=np.float32)
            H_cam2prj, mask1 = cv2.findHomography(src_points, dst_points, cv2.RANSAC, RANSAC_REPROJ)
            # 反向
            H_prj2cam, mask2 = cv2.findHomography(dst_points, src_points, cv2.RANSAC, RANSAC_REPROJ)
            if (H_cam2prj is None) or (H_prj2cam is None):
                continue
            in1 = int(np.sum(mask1)) if mask1 is not None else len(src_points)
            in2 = int(np.sum(mask2)) if mask2 is not None else len(src_points)
            if ((in1 / len(src_points)) < MIN_INLIER_RATIO) or ((in2 / len(src_points)) < MIN_INLIER_RATIO):
                continue
            corner_h = np.array([[cx_f, cy_f, 1.0]], dtype=np.float32).T
            proj_homo = H_cam2prj @ corner_h
            if abs(proj_homo[2, 0]) < 1e-6:
                continue
            u_proj = float(proj_homo[0, 0] / proj_homo[2, 0])
            v_proj = float(proj_homo[1, 0] / proj_homo[2, 0])
            # 范围检测
            if not (0 <= u_proj < proj_w and 0 <= v_proj < proj_h):
                continue
            prj_homo = np.array([[u_proj, v_proj, 1.0]], dtype=np.float32).T
            cam_back = H_prj2cam @ prj_homo
            if abs(cam_back[2, 0]) < 1e-6:
                continue
            cx_b = cam_back[0, 0] / cam_back[2, 0]
            cy_b = cam_back[1, 0] / cam_back[2, 0]
            e = math.hypot(cx_b - cx_f, cy_b - cy_f)
            if e > CONSISTENCY_ERR:
                continue
            proj_pts.append([u_proj, v_proj])
            obj_pts.append(objp_world[idx])
            cam_pts.append([cx_f, cy_f])
        print(f"[INFO] 图片 {number}: 原始角点={len(imgp_cam)}, 匹配成功={len(proj_pts)}")
        if len(proj_pts) < 40:
            print(f"[WARN] 图片 {number}: 有效点不足，跳过本图")
            continue
        proj_objpoints.append(np.float32(obj_pts).reshape(-1,1,3))
        proj_imgpoints.append(np.float32(proj_pts).reshape(-1,1,2))
        stereo_imgpoints_cam.append(np.float32(cam_pts).reshape(-1,1,2))
    return proj_objpoints,proj_imgpoints,stereo_imgpoints_cam
def calib_projector(proj_imgpoints_list, cam_imgpoints_list, objpoints_list,
                    projector_size=(854,480), cam_size=(1440,1080)):
    flags_proj = (cv2.CALIB_ZERO_TANGENT_DIST |
                  cv2.CALIB_FIX_K3 |
                  cv2.CALIB_FIX_K4 |
                  cv2.CALIB_FIX_K5 )
    # flags_proj = cv2.CALIB_FIX_INTRINSIC
    ret_proj, K_proj, D_proj, rvecs_p, tvecs_p = cv2.calibrateCamera(
        objpoints_list,            # 多视图 obj
        proj_imgpoints_list,       # 多视图 projector 点
        projector_size,
        None, None,
        flags=flags_proj
    )
    print("K_proj",K_proj )
    print("D_proj",D_proj )
    print(f"[INFO] 投影仪初始标定误差（RMS）= {ret_proj:.4f}")

    flags_stereo = cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_FOCAL_LENGTH
    # flags_stereo = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, _, _, K_proj2, D_proj2, R, T, _, _ = cv2.stereoCalibrate(
        objpoints_list,              # 世界点（多视图）
        cam_imgpoints_list,          # 相机像素点（多视图）
        proj_imgpoints_list,         # 投影仪像素点（多视图）
        K_cam, D_cam,                # 已知相机内参
        K_proj, D_proj,              # projector 初始内参
        cam_size,flags=flags_stereo,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5))
    print(f"[INFO] stereoCalibrate 优化误差（RMS）= {ret_stereo:.4f}")

    R_flip = np.array([[-1,0,0],[0,-1,0],[0,0,1]], float)
    R_fixed = R_flip @ R
    T_fixed = R_flip @ T
    return K_proj2, D_proj2, R_fixed, T_fixed


if __name__ == "__main__":
    """
    有效工作距离大概是70cm左右，相机与投影仪在同一平面，二者外壳相距7.5cm左右
    主要还是看二者的相对姿态。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    pattern = (11,8)
    square_size = 15  # mm
    # 1) 相机标定（保持你原来的函数）
    cam_params = calibrate_camera("images/calib/camera", square_size, pattern)
    K_cam, D_cam, mm_per_px = cam_params[1], cam_params[2], cam_params[8]

    # 全局容器：每个元素是单张棋盘图对应的一组角点（proj/cam/obj）
    proj_objpoints_list = []
    proj_imgpoints_list = []
    stereo_imgpoints_cam_list = []

    for i in [1,2,3,4,5,6]:
        gray_files = []
        graycode_dir = f"images/calib/p{i}"
        black_img = cv2.imread(f"images/frame_black_{i}.png", cv2.IMREAD_GRAYSCALE)
        white_img = cv2.imread(f"images/frame_white_{i}.png", cv2.IMREAD_GRAYSCALE)
        gray_paths = get_sorted_gray_paths(graycode_dir)
        for f in gray_paths:
            filename = os.path.basename(f)
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            gray_files.append(img)

        # 2) 在 white_img 上检测棋盘（每个姿态对应一张白光棋盘图）
        ret_c, corners = cv2.findChessboardCorners(white_img, pattern, None)
        if not ret_c:
            print(f"⚠️ 姿态 {i} 的白光图未检测到棋盘，跳过")
            continue
        # 亚像素精修 这一步是投影仪对应的棋盘格角点
        gray_for_subpix = white_img.copy()
        corners2 = cv2.cornerSubPix(gray_for_subpix, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # 构建对应的 世界坐标
        objp = np.zeros((pattern[0]*pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
        objp *= square_size

        proj_grey_h, proj_grey_v, mask_grey = decode_graycode_full(gray_files, black_img, white_img, (854,480))
        # show("proj_grey_h", proj_grey_h)
        # proj_phase_h, proj_phase_v = decode_phase()
        # phi_wrapped_h = gray_guided_phase_refine(
        #     gray_coord=proj_grey_h,
        #     phi_wrapped=proj_phase_h,
        #     width=854,
        #     freq=37,
        #     max_diff=3.0
        # )
        #
        # phi_wrapped_v = gray_guided_phase_refine(
        #     gray_coord=proj_grey_v,
        #     phi_wrapped=proj_phase_v,
        #     width=480,
        #     freq=37,
        #     max_diff=3.0
        # )
        #
        # proj_h, proj_v = fuse_gray_phase_keep_gray(
        #     proj_grey_h,
        #     proj_grey_v,
        #     phi_wrapped_h,
        #     phi_wrapped_v,
        #     max_diff=5.0
        # )
        # show("proj_h",proj_h)

        # show_histogram("proj_grey_h",proj_grey_h)

        # 5) 从单姿态的 proj_gray 与该姿态的相机角点 计算投影仪角点（calib_points 设计会返回 list）
        (proj_objpoints_pose, proj_imgpoints_pose,stereo_imgpoints_cam_pose) = calib_points(
            [objp], [corners2], proj_grey_h, proj_grey_v,i)
        for o, p, c in zip(proj_objpoints_pose, proj_imgpoints_pose, stereo_imgpoints_cam_pose):
            proj_objpoints_list.append(o)       # o shape: (M,1,3)
            proj_imgpoints_list.append(p)       # p shape: (M,1,2)
            stereo_imgpoints_cam_list.append(c) # c shape: (M,1,2)
        print(f"[pose {i}] 收集到 proj corners sets: {len(proj_objpoints_pose)}")
    # 融合：如果没有任何有效姿态就退出
    if len(proj_imgpoints_list) == 0:
        raise RuntimeError("未收集到任何姿态的有效角点，无法标定")
    # 6) 最后标定投影仪并做 stereoCalibrate
    # np.save("K_cam.npy", K_cam)
    # np.save("D_cam.npy", D_cam)
    # np.save("proj_imgpoints_list.npy",proj_imgpoints_list)
    # np.save("stereo_imgpoints_cam_list.npy", stereo_imgpoints_cam_list)
    # np.save("proj_objpoints_list.npy", proj_objpoints_list)
    # exit()
# if __name__ == "__main__":
#     proj_imgpoints_list = np.load("proj_imgpoints_list.npy")
#     stereo_imgpoints_cam_list = np.load("stereo_imgpoints_cam_list.npy")
#     proj_objpoints_list = np.load("proj_objpoints_list.npy")
#     K_cam = np.load("K_cam.npy")
#     D_cam = np.load("D_cam.npy")

    K_proj, D_proj, R_cam2prj, T_cam2prj = calib_projector(proj_imgpoints_list, stereo_imgpoints_cam_list,
                proj_objpoints_list,projector_size=(854,480), cam_size=(1440,1080))

    # 输出与保存
    print("外参 R_cam2prj:", R_cam2prj)
    print("外参 T_cam2prj:", T_cam2prj)
    print("内参 K_proj:", K_proj)
    print("畸变 D_proj:", D_proj)
    print("畸变 D_cam:", D_cam)
    print("畸变 K_cam:", K_cam)

    rvec, _ = cv2.Rodrigues(R_cam2prj)
    angle_deg = np.degrees(np.linalg.norm(rvec))
    print(f"相机-投影仪夹角: {angle_deg:.2f}°")
    print(np.linalg.det(R_cam2prj))
    print(f"📐 相机-投影仪基线长度: {np.linalg.norm(T_cam2prj):.2f} mm")

    np.savez("calibration_params_gray.npz",
             K_cam=K_cam, D_cam=D_cam,
             K_proj=K_proj, D_proj=D_proj,
             R=R_cam2prj, T=T_cam2prj)


