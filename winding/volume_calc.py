import numpy as np
import open3d as o3d

import numpy as np
from matplotlib import pyplot as plt


def project_to_axis(points: np.ndarray, center: np.ndarray, axis: np.ndarray):
    """
    投影到主轴坐标系
    返回:
        z: 轴向坐标
        r: 到轴线距离
    """
    X = points - center
    z = X @ axis
    foot = np.outer(z, axis)
    rv = X - foot
    r = np.linalg.norm(rv, axis=1)
    return z, r
def build_slice_stats(z, r, n_slices=300, min_points_per_slice=60):
    z_min, z_max = np.min(z), np.max(z)
    edges = np.linspace(z_min, z_max, n_slices + 1)

    centers = []
    counts = []
    q10 = []
    q20 = []
    q30 = []
    q50 = []
    q70 = []
    q90 = []
    q95 = []

    for i in range(n_slices):
        mask = (z >= edges[i]) & (z < edges[i + 1])
        rr = r[mask]
        centers.append((edges[i] + edges[i + 1]) * 0.5)
        counts.append(rr.size)

        if rr.size < min_points_per_slice:
            q10.append(np.nan)
            q20.append(np.nan)
            q30.append(np.nan)
            q50.append(np.nan)
            q70.append(np.nan)
            q90.append(np.nan)
            q95.append(np.nan)
            continue

        q10.append(np.percentile(rr, 10))
        q20.append(np.percentile(rr, 20))
        q30.append(np.percentile(rr, 30))
        q50.append(np.percentile(rr, 50))
        q70.append(np.percentile(rr, 70))
        q90.append(np.percentile(rr, 90))
        q95.append(np.percentile(rr, 90))

    stats = {
        "z_min": float(z_min),
        "z_max": float(z_max),
        "centers": np.asarray(centers),
        "counts": np.asarray(counts),
        "q10": np.asarray(q10),
        "q20": np.asarray(q20),
        "q30": np.asarray(q30),
        "q50": np.asarray(q50),
        "q70": np.asarray(q70),
        "q90": np.asarray(q90),
        "q95": np.asarray(q95),
    }
    return stats


def find_stable_end_radius(stats,height,side="top",
    search_ratio=0.04,edge_exclude_ratio=0.0,band_slice_half_width=1,):
    centers = stats["centers"]
    z_min = stats["z_min"]
    z_max = stats["z_max"]
    counts = stats["counts"]

    q90 = stats["q90"]
    q95 = stats["q95"]

    if height <= 0:
        raise ValueError("点云高度异常")

    if side == "top":
        mask_search = (
            (centers >= z_max - search_ratio * height) &
            (centers <= z_max - edge_exclude_ratio * height)
        )
    else:
        mask_search = (
            (centers <= z_min + search_ratio * height) &
            (centers >= z_min + edge_exclude_ratio * height)
        )

    valid = (
        mask_search &
        np.isfinite(q90) &
        np.isfinite(q95) &
        (counts > 0)
    )

    idx = np.where(valid)[0]
    if len(idx) == 0:
        raise ValueError(f"{side} 端未找到有效切片，请调大 search_ratio 或降低 min_points_per_slice")

    score = []
    for i in idx:
        left = max(0, i - 1)
        right = min(len(centers), i + 2)

        local = q95[left:right]
        local = local[np.isfinite(local)]

        if len(local) == 0:
            score.append(np.inf)
            continue

        stability = np.std(local)

        # 找裸露筒子的外半径：
        # 半径要偏大，但局部不能乱跳
        s = -q95[i] + 2.0 * stability
        score.append(s)

    best_idx = idx[np.argmin(score)]

    left = max(0, best_idx - band_slice_half_width)
    right = min(len(centers), best_idx + band_slice_half_width + 1)
    local_idx = np.arange(left, right)

    local_valid = (
        np.isfinite(q90[local_idx]) &
        np.isfinite(q95[local_idx]) &
        (counts[local_idx] > 0)
    )

    local_idx = local_idx[local_valid]

    if len(local_idx) == 0:
        local_idx = np.array([best_idx])

    # 裸露筒子外半径，取 q95 为主
    radius = np.median(q95[local_idx])
    z_ref = np.median(centers[local_idx])

    return {
        "slice_index": int(best_idx),
        "z_ref": float(z_ref),
        "radius": float(radius),
        "diameter": float(2.0 * radius),
        "num_slices_used": int(len(local_idx)),
    }
def measure_height(points, axis_point, axis_dir, q_bottom=0.01, q_top=0.99):

    points = np.asarray(points.points, dtype=np.float64)
    axis_point = np.asarray(axis_point, dtype=np.float64).reshape(3)
    axis_dir = np.asarray(axis_dir, dtype=np.float64).reshape(3)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是形状 (N, 3) 的数组")
    if len(points) == 0:
        raise ValueError("points 不能为空")
    if not (0.0 <= q_bottom < q_top <= 1.0):
        raise ValueError("必须满足 0 <= q_bottom < q_top <= 1")

    norm_axis = np.linalg.norm(axis_dir)
    if norm_axis < 1e-12:
        raise ValueError("axis_dir 长度过小，无法归一化")
    axis_dir = axis_dir / norm_axis
    t = (points - axis_point) @ axis_dir
    t_bottom = np.quantile(t, q_bottom)
    t_top = np.quantile(t, q_top)
    height = t_top - t_bottom

    return height
def measure_cone_core(point_cloud,n_slices=300,min_points_per_slice=60,
    search_ratio=0.18,edge_exclude_ratio=0.01,band_slice_half_width=3,
):

    if isinstance(point_cloud, str):
        print("检查输入文件")
    else:
        point_cloud = np.asarray(point_cloud.points)
        points = np.asarray(point_cloud, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point_cloud 必须是 Nx3 数组或点云文件路径")
    if len(points) < 100:
        raise ValueError("点云点数太少")
    # pcd = make_point_cloud(points)
    # center_mesh = make_center_sphere(axis_point, radius=2.0, color=(1, 0, 0))  # 红色中心点
    # axis_line = make_axis_line(axis_point, axis_dir, length=300.0, color=(0, 1, 0))  # 绿色轴线
    # o3d.visualization.draw_geometries([pcd, center_mesh, axis_line])
    z, r = project_to_axis(points, axis_point, axis_dir)

    # 2) 切片统计
    stats = build_slice_stats(z, r, n_slices=n_slices, min_points_per_slice=min_points_per_slice)
    height = measure_height(cloud, axis_point, axis_dir, q_bottom=0.01, q_top=0.99)

    # 3) 顶部/底部裸露筒子半径
    top = find_stable_end_radius(
        stats,
        height,
        side="top",
        search_ratio=search_ratio,
        edge_exclude_ratio=edge_exclude_ratio,
        band_slice_half_width=band_slice_half_width,
    )

    bottom = find_stable_end_radius(
        stats,
        height,
        side="bottom",
        search_ratio=search_ratio,
        edge_exclude_ratio=edge_exclude_ratio,
        band_slice_half_width=band_slice_half_width,
    )

    # 5) 圆台体积
    r_top = top["radius"]
    r_bottom = bottom["radius"]
    volume = np.pi * height * (r_top * r_top + r_top * r_bottom + r_bottom * r_bottom) / 3.0

    return {
        "height": float(height),
        "top_radius": float(r_top),
        "top_diameter": float(2.0 * r_top),
        "bottom_radius": float(r_bottom),
        "bottom_diameter": float(2.0 * r_bottom),
        "volume": float(volume),
        "top_z": float(top["z_ref"]),
        "bottom_z": float(bottom["z_ref"]),
        "top_detail": top,
        "bottom_detail": bottom,
    }

#计算总体积
def calc_total_volume(points,axis_dir,axis_point,n_slices,radius_quantile,
                      min_points_per_slice=40,smooth_window=5):

    points = np.asarray(points.points, dtype=np.float64)
    axis_dir = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    axis_point = np.asarray(axis_point, dtype=np.float64).reshape(3)
    norm_axis = np.linalg.norm(axis_dir)
    axis_dir = axis_dir / norm_axis
    vec = points - axis_point
    t = vec @ axis_dir
    foot = axis_point + np.outer(t, axis_dir)
    r = np.linalg.norm(points - foot, axis=1)
    t_min = np.min(t)
    t_max = np.max(t)

    edges = np.linspace(t_min, t_max, n_slices + 1)
    centers = []
    radii = []
    for i in range(n_slices):
        t0, t1 = edges[i], edges[i + 1]
        if i < n_slices - 1:
            mask = (t >= t0) & (t < t1)
        else:
            mask = (t >= t0) & (t <= t1)
        if np.count_nonzero(mask) < min_points_per_slice:
            continue
        r_slice = r[mask]
        r_outer = np.quantile(r_slice, radius_quantile)
        centers.append((t0 + t1) * 0.5)
        radii.append(r_outer)
    if len(radii) < 5:
        raise ValueError("有效切片太少，无法稳定计算总体积")
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)

    if smooth_window is not None and smooth_window > 1 and len(radii) >= smooth_window:
        k = int(smooth_window)
        if k % 2 == 0:
            k += 1
        pad = k // 2
        radii_pad = np.pad(radii, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=np.float64) / k
        radii = np.convolve(radii_pad, kernel, mode="valid")

    areas = np.pi * radii ** 2
    volume = np.trapz(areas, centers)

    return float(volume)

if __name__ == "__main__":
    ply_path = "../cloud_rebuild/whole_clouds/ICP_cloud_cusha_duibi.ply"
    axis_point = np.load("../cloud_rebuild/center_small.npy")

    d = np.load("../cloud_rebuild/turntable_axis.npz")
    axis_dir = d["axis_dir"]
    cloud = o3d.io.read_point_cloud(ply_path)
    result = measure_cone_core(cloud, n_slices=600,min_points_per_slice=25,
        search_ratio=0.02,edge_exclude_ratio=0.0,band_slice_half_width=1,)
    print("===== 圆台空筒子测量结果 =====")
    print(f"整体高度: {result['height']:.4f}")
    print(f"上端半径: {result['top_radius']:.4f}")
    print(f"上端直径: {result['top_diameter']:.4f}")
    print(f"下端半径: {result['bottom_radius']:.4f}")
    print(f"下端直径: {result['bottom_diameter']:.4f}")
    print(f"空筒子总体积 = {result['volume'] / 1000:.2f} 立方厘米")
    volume_all = calc_total_volume(
        points=cloud,axis_dir=axis_dir,axis_point=axis_point,
        n_slices=600,radius_quantile=0.95, min_points_per_slice=500, smooth_window=2)
    print(f"点云总体积 = {volume_all / 1000:.2f} 立方厘米")

    # volume_yarn = volume_all - result['volume']
    # print(f"纱线总体积 = {volume_yarn / 1000:.1f} 立方厘米")

