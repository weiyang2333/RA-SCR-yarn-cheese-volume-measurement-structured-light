
import os
import numpy as np
import open3d as o3d


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
    return out, mask

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
        return [n0, n1]
    else:
        return [n0]
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


def compute_axis_point_least_squares(axis_dir, all_points):

    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 收集所有点
    X = np.vstack(all_points)

    # 每个点到轴的“垂直分量算子”
    # P = I - d d^T
    I = np.eye(3)
    P = I - np.outer(axis_dir, axis_dir)
    PX = (P @ X.T).T
    p_perp = PX.mean(axis=0)

    axis_point = p_perp
    return axis_point

def make_axis_line(axis, point, length=400,radius=2):
    axis = axis / np.linalg.norm(axis)
    p0 = point - axis * length
    p1 = point + axis * length
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([p0, p1])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    #显示point
    color = [1, 0, 0]
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(point)

    #显示dir
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    points = []
    lines = []
    idx = 0
    dash = 10
    gap = 6
    t = -length
    while t < length:
        p_start = point + axis * t
        p_end = point + axis * min(t + dash, length)
        points.append(p_start)
        points.append(p_end)
        lines.append([idx, idx + 1])
        idx += 2
        t += dash + gap
    line_dir = o3d.geometry.LineSet()
    line_dir.points = o3d.utility.Vector3dVector(points)
    line_dir.lines = o3d.utility.Vector2iVector(lines)
    line_dir.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(lines))
    return line,sphere,line_dir
#########




if __name__ == "__main__":
    # 读取 PLY 文件
    could_path = "axis_clouds\he_feiyuanxin"
    could_file = ["cloud_0.ply", "cloud_1.ply", "cloud_2.ply", "cloud_2.ply", "cloud_4.ply", "cloud_5.ply",
                  "cloud_6.ply", "cloud_7.ply", "cloud_8.ply", "cloud_9.ply", "cloud_10.ply", "cloud_11.ply"]
    #这个角度是你转动圆台的角度 根据实际情况进行调整 理想情况 12次就是 360/12，依此填充角度即可
    angel_dir = {0: 0, 1: 40.75, 2: 67.62, 3: 91.2, 4: 123.12, 5: 149.99, 6: 180.14, 7: 221.86,
                 8: 248.57, 9: 271.69, 10: 302.89, 11: 330.17}
    clouds = []
    clouds_dir = []
    yuan_big = []
    all_normals = []  # 收集所有帧的 side normals
    all_points = []
    all_plane_centers = []  # 新增：每一帧一个或两个平面中心
    for i in could_file:
        file_path = os.path.join(could_path, i)
        pcd = o3d.io.read_point_cloud(file_path)
        pc_obj, _ = crop_pcd_xyz(pcd, y=(0, 87)) #对点云执行裁剪去掉圆台点云
        # o3d.visualization.draw_geometries([pc_obj])
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
    axis_dir = axis_dir_from_normals(all_normals)
    print("axis_dir:",axis_dir)
    axis_point = compute_axis_point_least_squares(axis_dir, all_points)   #仅作示范作用 不启用

    print("\n=== RANSAC circle centers on turntable layers ===")

    np.savez(
        "turntable_axis.npz",
        axis_dir=axis_dir.astype(np.float64),
        axis_point=axis_point.astype(np.float64),
    )