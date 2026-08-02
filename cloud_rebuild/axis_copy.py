import copy
import os
import re
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import open3d as o3d
# from cloud_model import *

def crop_pcd_xyz(
    pcd: o3d.geometry.PointCloud,
    x=None, y=None, z=None,
    invert: bool = False,
):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd, np.zeros((0,), dtype=bool)
    mask = np.ones((len(pts),), dtype=bool)
    def apply_axis(axis_vals, rng):
        nonlocal mask
        if rng is None:
            return
        lo, hi = rng
        if lo is not None:
            mask &= (axis_vals >= lo)
        if hi is not None:
            mask &= (axis_vals <= hi)

    apply_axis(pts[:, 0], x)
    apply_axis(pts[:, 1], y)
    apply_axis(pts[:, 2], z)

    if invert:
        mask = ~mask

    idx = np.where(mask)[0]
    out = pcd.select_by_index(idx)

    # 如果你想“保留属性一致性”，select_by_index 已经会把 colors/normals 一起带过去
    # copy_cloud 这里只是语义参数，Open3D本身会返回新对象
    return out, mask
#对点云做一次 RANSAC 平面拟合
#在 origin 处创建一个沿 direction 方向的箭头（LineSet）
def create_arrow(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float = 50.0,
    color=(1.0, 0.0, 0.0),
):
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    end = origin + direction * length
    points = [origin, end]
    lines = [[0, 1]]
    colors = [color]
    arrow = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    arrow.colors = o3d.utility.Vector3dVector(colors)
    return arrow
def ransac_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold=1.5,
    ransac_n=3,
    num_iterations=1000,
):

    if len(pcd.points) < 50:
        return None, None, None
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inlier_cloud, outlier_cloud
#从一帧点云中提取 1~2 个 dominant side 平面
def extract_side_planes_by_ransac(
    pcd: o3d.geometry.PointCloud,
    max_planes=2,
    min_points=300,
    min_cloud_points=800,
    distance_threshold=1.5,
):
    planes = []   # [(n, center, inlier_points)]
    if len(pcd.points) < min_cloud_points:
        return planes
    rest = pcd
    for _ in range(max_planes):
        if len(rest.points) < min_cloud_points:
            break
        plane_model, inlier, outlier = ransac_plane(
            rest,
            distance_threshold=distance_threshold,
        )
        if plane_model is None:
            break
        if len(inlier.points) < min_points:
            break
        n = np.array(plane_model[:3], dtype=np.float64)
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        center = np.mean(np.asarray(inlier.points), axis=0)
        # ★ 把 inlier.points 一起返回
        planes.append((n, center, np.asarray(inlier.points)))
        rest = outlier
    return planes
#对聚合侧面簇进行划分 以45度为基准
def filter_planes_by_angle(
    normals,
    angle_threshold_deg=45.0,
):
    if len(normals) <= 1:
        return normals
    n0, n1 = normals[:2]
    cosang = np.clip(abs(n0 @ n1), 0.0, 1.0)
    ang = np.degrees(np.arccos(cosang))

    if ang > angle_threshold_deg:
        return [n0, n1]   # 两个侧面
    else:
        return [n0]       # 实际是同一个侧面
#根据所有帧收集到的 side normals，使用 SVD / PCA 估计旋转轴方向 axis_dir
def axis_dir_from_normals(all_normals):
    all_normals = np.asarray(all_normals, dtype=np.float64)

    if all_normals.ndim != 2 or all_normals.shape[1] != 3:
        raise ValueError("all_normals 维度必须是 (N, 3)")
    if all_normals.shape[0] < 2:
        raise ValueError("法向数量过少，无法估计 axis_dir")
    # 单位化（保险）
    all_normals /= np.linalg.norm(all_normals, axis=1, keepdims=True)
    # SVD / PCA
    _, _, vh = np.linalg.svd(all_normals, full_matrices=False)
    # 最小奇异值对应的方向
    axis_dir = vh[-1]
    axis_dir /= np.linalg.norm(axis_dir)
    # 固定符号，避免正反翻转造成“抖动感”
    if axis_dir[2] < 0:
        axis_dir = -axis_dir
    return axis_dir

def optimize_axis_dir_by_radius_std_filtered(
    axis_dir_init,
    all_points,
    std_keep_threshold=3.0,
    min_keep_frames=4,
    angle_deg_range=6.0,
    angle_deg_step=0.5,
):
    """
    基于“先筛帧、再优化”的 axis_dir 优化函数
    - 自动剔除误差大的帧
    - 只用“好帧”拉 axis_dir
    """

    axis_dir_init = axis_dir_init / np.linalg.norm(axis_dir_init)

    # ===============================
    # Step 1: 计算每帧 radius std
    # ===============================
    axis_point_init = axis_point_from_normals(all_points, axis_dir_init)

    frame_stds = []
    for i, pts in enumerate(all_points):
        if pts.shape[0] < 50:
            continue

        vec = pts - axis_point_init
        proj = vec @ axis_dir_init
        perp = vec - np.outer(proj, axis_dir_init)
        radii = np.linalg.norm(perp, axis=1)

        frame_stds.append((i, np.std(radii)))

    if len(frame_stds) == 0:
        raise RuntimeError("No valid frames for axis_dir optimization.")

    # ===============================
    # Step 2: 筛选“好帧”
    # ===============================
    kept_ids = [
        i for (i, std) in frame_stds
        if std < std_keep_threshold
    ]

    # 保底：至少保留 min_keep_frames 个
    if len(kept_ids) < min_keep_frames:
        frame_stds_sorted = sorted(frame_stds, key=lambda x: x[1])
        kept_ids = [i for (i, _) in frame_stds_sorted[:min_keep_frames]]

    kept_points = [all_points[i] for i in kept_ids]

    print("\n[optimize_axis_dir_by_radius_std_filtered]")
    print(f"kept frames: {kept_ids}")

    # ===============================
    # Step 3: 用“好帧”优化 axis_dir
    # ===============================
    u, v = make_uv(axis_dir_init)

    best_dir = axis_dir_init
    best_score = np.inf
    best_point = None
    records = []

    for du_deg in np.arange(-angle_deg_range, angle_deg_range + 1e-9, angle_deg_step):
        for dv_deg in np.arange(-angle_deg_range, angle_deg_range + 1e-9, angle_deg_step):

            du = np.deg2rad(du_deg)
            dv = np.deg2rad(dv_deg)

            axis_dir = axis_dir_init + du * u + dv * v
            axis_dir /= np.linalg.norm(axis_dir)

            try:
                axis_point = axis_point_from_normals(kept_points, axis_dir)
            except Exception:
                continue

            radial_stds = []
            for pts in kept_points:
                if pts.shape[0] < 50:
                    continue
                vec = pts - axis_point
                proj = vec @ axis_dir
                perp = vec - np.outer(proj, axis_dir)
                radii = np.linalg.norm(perp, axis=1)
                radial_stds.append(np.std(radii))

            if len(radial_stds) == 0:
                continue

            score = float(np.mean(radial_stds))
            records.append({"du": du_deg, "dv": dv_deg, "score": score})

            if score < best_score:
                best_score = score
                best_dir = axis_dir
                best_point = axis_point

    print(f"best mean radius std (filtered) = {best_score:.4f}")

    return best_dir, best_point, best_score, records

###########################################point
def make_uv(axis_dir):
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    if abs(axis_dir[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])

    u = np.cross(axis_dir, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(axis_dir, u)
    return u, v
def fit_circle_2d(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c[:2]   # (cx, cy)
def axis_point_from_normals(all_frame_points, axis_dir):
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    # 构造垂直基
    u, v = make_uv(axis_dir)
    centers_3d = []
    for pts in all_frame_points:
        if pts.shape[0] < 50:
            continue
        # 投影到 2D
        pts_2d = np.column_stack([
            pts @ u,
            pts @ v
        ])
        # 拟合该帧圆心
        center_2d = fit_circle_2d(pts_2d)
        # 反投影回 3D（轴在该平面里的投影）
        center_3d = center_2d[0] * u + center_2d[1] * v
        centers_3d.append(center_3d)
    if len(centers_3d) == 0:
        raise RuntimeError("No valid frame centers for axis_point estimation.")
    # 所有帧圆心统一
    axis_point = np.mean(np.vstack(centers_3d), axis=0)
    return axis_point

def compute_axis_point_least_squares(axis_dir, all_points):
    """
    给定 axis_dir，用最小二乘意义求 axis_point
    适用于：长方体 / 非圆心 / 平面侧面点
    """

    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 收集所有点
    X = np.vstack(all_points)

    # 每个点到轴的“垂直分量算子”
    # P = I - d d^T
    I = np.eye(3)
    P = I - np.outer(axis_dir, axis_dir)

    # 我们要求：min_p Σ ||P (x_i - p)||^2
    # => p = argmin ||P X - P p||^2
    # => 在 P 空间中，p 是投影点的质心

    PX = (P @ X.T).T
    p_perp = PX.mean(axis=0)

    # 找一个满足：P p = p_perp 的 p
    # 即 p = p_perp + t * d （任意 t 都行）
    axis_point = p_perp

    return axis_point

#*******************************

def check_axis_dir_for_reconstruction_subset(
    axis_dir,
    all_points,
    axis_point,
    subset_ids=None,
):
    print("\n========== Check axis_dir ==========")
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # -------- 选择要检查的索引 --------
    if subset_ids is None:
        ids = range(len(all_points))
        print("[mode] full set")
    else:
        ids = subset_ids
        print(f"[mode] subset: {subset_ids}")
    radial_stds = []
    for idx in ids:
        pts = all_points[idx]
        if pts.shape[0] < 50:
            continue
        vec = pts - axis_point
        proj = vec @ axis_dir
        perp = vec - np.outer(proj, axis_dir)
        radii = np.linalg.norm(perp, axis=1)
        std_r = np.std(radii)
        radial_stds.append(std_r)
        print(f"[frame {idx:02d}] radius std = {std_r:.4f}")
    if len(radial_stds) == 0:
        print("No valid frames for checking.")
        return np.inf
    mean_std = np.mean(radial_stds)
    print("-----------------------------------------------")
    print(f"mean radius std = {mean_std:.4f}")

    return mean_std

if __name__ == "__main__":
    # 读取 PLY 文件
    could_path = "axis_clouds"
    could_file = ["cloud_0.ply","cloud_1.ply","cloud_2.ply","cloud_2.ply","cloud_4.ply","cloud_5.ply",
                  "cloud_6.ply","cloud_7.ply","cloud_8.ply","cloud_9.ply","cloud_10.ply","cloud_11.ply"]
    #  10 11
    angel_dir = {0: 0,1: 40.75,2: 67.62,3: 91.2,4: 123.12,5: 149.99,6: 180.14,7: 221.86,
        8: 248.57,9: 271.69,10: 302.89,11: 330.17}
    clouds = []
    clouds_dir = []
    arrow_length = 50.0
    all_normals = []  # 收集所有帧的 side normals
    all_points = []
    all_plane_centers = []  # 新增：每一帧一个或两个平面中心
    for i in could_file:
        file_path = os.path.join(could_path, i)
        pcd = o3d.io.read_point_cloud(file_path)
        pc_obj, _ = crop_pcd_xyz(pcd, y=(5, 60)) #对点云执行裁剪去掉圆台点云
        # pc_obj = keep_largest_cluster(pc_obj, eps=6.0, min_points=80)
        # pc_obj = keep_main_radius(pc_obj, keep_ratio=0.95)
        # o3d.visualization.draw_geometries([pc_obj])
        # clouds.append(pcd)
        clouds_dir.append((i, pc_obj))
        planes = extract_side_planes_by_ransac(pc_obj,max_planes=2,
        min_points=300,min_cloud_points=6000,distance_threshold=1.5,)
        normals = [n for (n, center, pts) in planes]
        side_normals = filter_planes_by_angle(normals,angle_threshold_deg=45.0,)
        for (n, center, pts) in planes:
            # 只收集通过 45° 筛选的侧面对应的点
            for n_keep in side_normals:
                if abs(n @ n_keep) > 0.999:
                    all_normals.append(n)
                    all_points.append(pts)
                    all_plane_centers.append(center)
                    break
        # geoms = [pc_obj]
        # for (n, center) in planes:
        #     arrow = create_arrow(origin=center,direction=n,length=arrow_length,color=(1.0, 0.0, 0.0),)
        #     geoms.append(arrow)
        # o3d.visualization.draw_geometries(geoms)
        # print(f"[{i}] detected side planes = {len(side_normals)}")
        # for k, n in enumerate(side_normals):
        #     print(f"    side{k} normal = {n}")

    axis_dir = axis_dir_from_normals(all_normals)
    print("axis_dir:",axis_dir)
    # axis_dir_opt, axis_point_opt, score, records = optimize_axis_dir_by_radius_std_filtered(
    #     axis_dir_init=axis_dir,
    #     all_points=all_points,
    #     std_keep_threshold=2.0,  # 可以试 2.5 / 3.0 / 3.5
    # )
    #
    # check_axis_dir_for_reconstruction_subset(axis_dir_opt, all_points, axis_point_opt)
    # axis_point = compute_axis_point_least_squares(axis_dir_opt,all_points,)
    axis_point = axis_point_from_normals(all_points, axis_dir)
    print("axis_point:", axis_point)
    # axis_dir, axis_point, best_val = refine_axis_dir_and_point_engineering(
    #     clouds_dir=clouds_dir,
    #     merged_sides_axis_point=all_points,
    #     axis_dir0=axis_dir,
    #     angel_dir=angel_dir,
    #     angle_deg_range=0.5,
    #     angle_deg_step=0.1
    # )

    # print("refined axis_dir:", axis_dir)
    # print("refined axis_point:", axis_point)
    # print("best objective:", best_val)


    np.savez(
        "turntable_axis.npz",
        axis_dir=axis_dir.astype(np.float64),
        axis_point=axis_point.astype(np.float64),
    )