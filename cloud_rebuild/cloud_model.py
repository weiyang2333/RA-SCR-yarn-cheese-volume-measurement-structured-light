import copy
import os
import re
import pyvista as pv
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from types import SimpleNamespace
from sklearn.cluster import DBSCAN
import numpy as np
import open3d as o3d


# ---------------------------
# utils
# ---------------------------
#解算角度  调用看函数内第一行 示例为函数内第二行
def dms(d, m=0, s=0):
    # angle_deg = -dms(29, 24)
    # angel_dir = {i: -1 * i * 30 for i in range(12)}
    return d + m/60 + s/3600


#######创建轴显示
def create_full_axes(length=300.0, origin=(0, 0, 0), line_width_hint=None):
    """
    生成 6 条轴线：+X, -X, +Y, -Y, +Z, -Z （共 6 个）
    视觉上“瘦长”：用 LineSet 画线（Open3D原生线宽在不同后端可能不生效）
    """
    ox, oy, oz = origin
    L = float(length)

    # 7个点：原点 + 6个端点
    pts = np.array([
        [ox, oy, oz],        # 0: origin
        [ox + L, oy, oz],    # 1: +X
        [ox - L, oy, oz],    # 2: -X
        [ox, oy + L, oz],    # 3: +Y
        [ox, oy - L, oz],    # 4: -Y
        [ox, oy, oz + L],    # 5: +Z
        [ox, oy, oz - L],    # 6: -Z
    ], dtype=np.float64)

    # 6条线：都从原点连到端点
    lines = np.array([
        [0, 1],  # +X
        [0, 2],  # -X
        [0, 3],  # +Y
        [0, 4],  # -Y
        [0, 5],  # +Z
        [0, 6],  # -Z
    ], dtype=np.int32)

    # 颜色：X红、Y绿、Z蓝；负方向用稍暗一点（你也可以改成一样）
    colors = np.array([
        [1.0, 0.0, 0.0],   # +X
        [0.6, 0.0, 0.0],   # -X
        [0.0, 1.0, 0.0],   # +Y
        [0.0, 0.6, 0.0],   # -Y
        [0.0, 0.0, 1.0],   # +Z
        [0.0, 0.0, 0.6],   # -Z
    ], dtype=np.float64)

    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(pts)
    axis.lines = o3d.utility.Vector2iVector(lines)
    axis.colors = o3d.utility.Vector3dVector(colors)
    return axis

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
#####裁剪点云

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

#利用旋转轴裁剪点云
def crop_pcd_local(
    pcd: o3d.geometry.PointCloud,
    axis_point,
    axis_dir,
    x=None, y=None, z=None,
    invert: bool = False,
):
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        return pcd, np.zeros((0,), dtype=bool)

    axis_point = np.asarray(axis_point, dtype=np.float64).reshape(3)
    axis_dir = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    # 以 axis_dir 作为局部 Y 轴
    ey = axis_dir

    # 选一个不平行的参考向量
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, ey)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    ex = np.cross(ey, ref)
    ex = ex / (np.linalg.norm(ex) + 1e-12)

    ez = np.cross(ex, ey)
    ez = ez / (np.linalg.norm(ez) + 1e-12)

    # 投影到局部坐标
    vec = pts - axis_point
    x_local = vec @ ex
    y_local = vec @ ey
    z_local = vec @ ez

    mask = np.ones(len(pts), dtype=bool)

    def apply_axis(vals, rng):
        nonlocal mask
        if rng is None:
            return
        lo, hi = rng
        if lo is not None:
            mask &= (vals >= lo)
        if hi is not None:
            mask &= (vals <= hi)

    apply_axis(x_local, x)
    apply_axis(y_local, y)
    apply_axis(z_local, z)
    if invert:
        mask = ~mask
    idx = np.where(mask)[0]
    out = pcd.select_by_index(idx)

    return out
#####
# 使用 DBSCAN 根据点云距离自动分簇
def split_pointcloud_by_distance(pcd, eps=3.0, min_samples=20, min_points=1000):
    points = np.asarray(pcd.points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # 噪声
        pts = points[labels == label]
        if len(pts) < min_points:
            continue  # 点数太少，不保留
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        clusters.append(pc)
    clusters_s = sorted(clusters, key=lambda c: len(np.asarray(c.points)), reverse=True)
    return clusters_s
######
def make_uv(axis_dir):
    """
    根据轴方向 axis_dir，构造一组正交基 (u, v)，
    使得 u ⟂ v ⟂ axis_dir，且都是单位向量
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # 选一个不与 axis_dir 平行的参考向量
    if abs(axis_dir[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    # u = axis_dir × ref
    u = np.cross(axis_dir, ref)
    u = u / np.linalg.norm(u)

    # v = axis_dir × u
    v = np.cross(axis_dir, u)
    v = v / np.linalg.norm(v)

    return u, v

def fit_circle_2d_ransac(
    pts_2d,
    num_iter=2000,
    dist_thresh=1.5,
    min_inliers_ratio=0.3,
):
    pts = np.asarray(pts_2d)
    N = pts.shape[0]
    if N < 10:
        raise RuntimeError("Not enough points for circle RANSAC")

    best_inliers = []
    best_center = None

    for _ in range(num_iter):
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = pts[idx]

        # 三点共线就跳过
        area = np.cross(p2 - p1, p3 - p1)
        if abs(area) < 1e-6:
            continue

        # 解圆
        A = np.array([
            [2*(p2[0]-p1[0]), 2*(p2[1]-p1[1])],
            [2*(p3[0]-p1[0]), 2*(p3[1]-p1[1])]
        ])
        b = np.array([
            p2[0]**2 + p2[1]**2 - p1[0]**2 - p1[1]**2,
            p3[0]**2 + p3[1]**2 - p1[0]**2 - p1[1]**2
        ])

        try:
            center = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue

        r = np.linalg.norm(pts - center, axis=1)
        r0 = np.median(r)
        inliers = np.where(np.abs(r - r0) < dist_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_center = center

    if best_center is None or len(best_inliers) < min_inliers_ratio * N:
        raise RuntimeError("RANSAC circle fit failed")

    return best_center, best_inliers

#以下为ICP 建造
def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    return R
def transform_to_axis_frame(clouds, axis_dir, axis_point):
    """
    clouds: list[np.ndarray(N,3)] or list[o3d.geometry.PointCloud]
    """
    R = rotation_matrix_from_vectors(axis_dir, np.array([0, 0, 1]))
    axis_point = np.asarray(axis_point, dtype=float)

    clouds_z = []

    for P in clouds:
        # ✅ 统一转成 numpy
        if hasattr(P, "points"):  # Open3D PointCloud
            P_np = np.asarray(P.points)
        else:                     # numpy array
            P_np = np.asarray(P)

        P0 = P_np - axis_point
        Pz = (R @ P0.T).T
        clouds_z.append(Pz)

    return clouds_z, R


def apply_params_to_cloud(P, dx, dy, dtheta):
    c, s = np.cos(dtheta), np.sin(dtheta)
    Rot = np.array([[c, -s],
                    [s,  c]], dtype=np.float64)
    Q = P.copy()
    Q[:, :2] = Q[:, :2] @ Rot.T
    Q[:, 0] += dx
    Q[:, 1] += dy
    return Q




def assign_z_slices(P, z_min, z_max, num_sections):
    z = np.asarray(P)[:, 2]
    idx = ((z - z_min) / (z_max - z_min + 1e-12) * num_sections).astype(np.int32)
    return np.clip(idx, 0, num_sections - 1)


def get_global_z_range(clouds):
    valid_z = [np.asarray(P)[:, 2] for P in clouds if len(P) > 0]
    if len(valid_z) == 0:
        raise ValueError("点云为空，无法计算轴向范围")
    all_z = np.concatenate(valid_z)
    return float(np.min(all_z)), float(np.max(all_z))
def unpack_icp_params(x, K, num_sections=10):
    rigid = x[:3 * K].reshape(K, 3)
    cx = x[3 * K]
    cy = x[3 * K + 1]
    Rs = x[3 * K + 2: 3 * K + 2 + num_sections]
    return rigid, cx, cy, Rs

def transform_all_clouds_with_params(x, clouds_ds, K, num_sections=10):
    rigid, cx, cy, Rs = unpack_icp_params(x, K, num_sections)
    out = []
    for k in range(K):
        dx, dy, dtheta = rigid[k]
        out.append(apply_params_to_cloud(clouds_ds[k], dx, dy, dtheta))
    return out, cx, cy, Rs

def build_pair_correspondences_axis_icp(
    transformed_clouds,
    nn_dist_thresh,
    z_overlap_thresh,
    min_corr_points=20,
):
    """
    只建立相邻帧之间的对应:
    k <-> k+1
    """
    K = len(transformed_clouds)
    pairs = []

    for k in range(K):
        j = (k + 1) % K
        A = transformed_clouds[k]
        B = transformed_clouds[j]

        if len(A) < 10 or len(B) < 10:
            pairs.append((k, j, np.array([], dtype=int), np.array([], dtype=int)))
            continue

        tree_B = cKDTree(B)
        dists, idxs = tree_B.query(A, k=1)

        z_ok = np.abs(A[:, 2] - B[idxs, 2]) < z_overlap_thresh
        d_ok = dists < nn_dist_thresh
        keep = z_ok & d_ok

        src_idx = np.where(keep)[0]
        tgt_idx = idxs[keep]

        if len(src_idx) < min_corr_points:
            src_idx = np.array([], dtype=int)
            tgt_idx = np.array([], dtype=int)

        pairs.append((k, j, src_idx, tgt_idx))

    return pairs


def residuals_axis_icp_v2(
    x,
    pairs,
    clouds_ds,
    K,
    cyl_weight,
    pair_weight,
    smooth_weight,
    num_sections=10,
    radius_smooth_weight=0.2,
):
    res = []
    transformed, cx, cy, Rs = transform_all_clouds_with_params(
        x, clouds_ds, K, num_sections
    )
    rigid, _, _, _ = unpack_icp_params(x, K, num_sections)

    # 统一轴向范围：保证所有帧使用同一套 Rs(z) 分层
    z_min, z_max = get_global_z_range(transformed)

    # A. 相邻帧 ICP 主残差
    for (k, j, src_idx, tgt_idx) in pairs:
        if len(src_idx) == 0:
            continue

        A = transformed[k][src_idx]
        B = transformed[j][tgt_idx]
        diff = A - B
        res.append(pair_weight * diff.reshape(-1))

    # B. 分层共享截面先验：r - Rs(z)
    for k in range(K):
        P = transformed[k]
        if len(P) == 0:
            continue

        r = np.sqrt((P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2)

        # 去掉半径极端值，降低毛羽、缺口和离群点影响
        r_lo = np.quantile(r, 0.10)
        r_hi = np.quantile(r, 0.90)
        keep = (r >= r_lo) & (r <= r_hi)
        if np.count_nonzero(keep) < 20:
            keep = np.ones_like(r, dtype=bool)

        section_idx = assign_z_slices(P, z_min, z_max, num_sections)
        target_r = Rs[section_idx]

        res.append(cyl_weight * (r[keep] - target_r[keep]))

    # C. 每帧三自由度参数平滑项，减少单帧突跳造成切痕
    for k in range(K):
        k2 = (k + 1) % K
        dx1, dy1, th1 = rigid[k]
        dx2, dy2, th2 = rigid[k2]

        res.append(np.array([
            smooth_weight * (dx2 - dx1),
            smooth_weight * (dy2 - dy1),
            smooth_weight * 2.0 * (th2 - th1),
        ], dtype=np.float64))

    # D. 分层半径平滑项，避免相邻轴向层半径剧烈跳变
    for m in range(num_sections - 1):
        res.append(np.array([
            radius_smooth_weight * (Rs[m + 1] - Rs[m])
        ], dtype=np.float64))

    return np.concatenate(res) if len(res) > 0 else np.zeros((1,), dtype=np.float64)

def init_axis_icp_params(clouds_ds, num_sections=10):
    """
    初始化轴约束优化参数：
    [每帧 dx,dy,dtheta] + [cx,cy] + [R1...RM]

    其中 R1...RM 是按轴向分层统计得到的共享截面半径初值。
    """
    K = len(clouds_ds)
    params0 = []

    for _ in range(K):
        params0 += [0.0, 0.0, 0.0]

    # 共享截面中心初值
    params0 += [0.0, 0.0]

    z_min, z_max = get_global_z_range(clouds_ds)

    # 全局半径兜底值，避免某些层点太少
    global_r0 = float(np.median([
        np.median(np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2))
        for P in clouds_ds if len(P) > 0
    ]))

    Rs_init = []
    for m in range(num_sections):
        r_list = []

        for P in clouds_ds:
            if len(P) == 0:
                continue

            section_idx = assign_z_slices(P, z_min, z_max, num_sections)
            mask = section_idx == m

            if np.count_nonzero(mask) > 10:
                rr = np.sqrt(P[mask, 0] ** 2 + P[mask, 1] ** 2)
                r_list.append(float(np.median(rr)))

        if len(r_list) > 0:
            Rs_init.append(float(np.median(r_list)))
        else:
            Rs_init.append(global_r0)

    params0 += Rs_init
    return np.array(params0, dtype=np.float64)

def downsample_clouds_for_axis_icp(clouds_z, sample_step=2):
    clouds_ds = []
    for P in clouds_z:
        P = np.asarray(P, dtype=np.float64)
        if sample_step is not None and sample_step > 1:
            P = P[::sample_step]
        clouds_ds.append(P)
    return clouds_ds

"""
改进版轴约束 ICP + 分层共享截面半径 Rs(z)：
    1) 在轴坐标系下仅优化每帧 dx, dy, dtheta
    2) 使用相邻帧重叠区最近邻对应作为 ICP 主残差
    3) 将单一共享半径 R 改为轴向分层共享半径 Rs=[R1,...,RM]
    4) 加入相邻帧参数平滑项与分层半径平滑项
"""

def axis_constrained_icp_v2(
    clouds,axis_dir,axis_point,outer_iter=1,inner_max_nfev=50,sample_step=2,
    nn_dist_thresh=1.3,z_overlap_thresh=1.6,cyl_weight=0.04,pair_weight=2.0,
    smooth_weight=0.35,huber_f_scale=0.5,num_sections=10,radius_smooth_weight=0.2,
    verbose=True):

    clouds_z, R_axis = transform_to_axis_frame(clouds, axis_dir, axis_point)
    clouds_ds = downsample_clouds_for_axis_icp(clouds_z, sample_step=sample_step)
    K = len(clouds_ds)
    params = init_axis_icp_params(clouds_ds, num_sections=num_sections)
    last_cost = None
    success = True
    message = "ok"

    for it in range(outer_iter):
        transformed_now, _, _, _ = transform_all_clouds_with_params( params, clouds_ds, K, num_sections)
        pairs = build_pair_correspondences_axis_icp(transformed_now,nn_dist_thresh=nn_dist_thresh,
            z_overlap_thresh=z_overlap_thresh,min_corr_points=20)

        valid_pairs = sum(1 for (_, _, s, _) in pairs if len(s) > 0)
        total_corr = sum(len(s) for (_, _, s, _) in pairs)
        if total_corr < 50:
            success = False
            message = "有效对应点过少，ICP停止"
            break
        old_params = params.copy()
        result_ls = least_squares(
            lambda x: residuals_axis_icp_v2(
                x=x,pairs=pairs,clouds_ds=clouds_ds,
                K=K,cyl_weight=cyl_weight,pair_weight=pair_weight,
                smooth_weight=smooth_weight,num_sections=num_sections,
                radius_smooth_weight=radius_smooth_weight),
            params,loss="huber",f_scale=huber_f_scale,verbose=2 if verbose else 0,
            max_nfev=inner_max_nfev)

        params = result_ls.x
        last_cost = result_ls.cost
        # 收敛判据
        if result_ls.success and np.linalg.norm(params - old_params) < 1e-8:
            break

    aligned_clouds = []
    rigid, cx, cy, Rs = unpack_icp_params(params, K, num_sections)
    for k in range(K):
        dx, dy, dtheta = rigid[k]
        P = apply_params_to_cloud(clouds_z[k], dx, dy, dtheta)
        aligned_clouds.append(P)

    result = SimpleNamespace(
        x=params,cost=last_cost,success=success,message=message,
        cx=cx,cy=cy,Rs=Rs,
        R_axis=R_axis,
        num_sections=num_sections,
    )

    return aligned_clouds, result

def move_back_axis(aligned_clouds, cx, cy):
    out = []
    shift = np.array([-cx, -cy, 0.0], dtype=np.float64)
    for P in aligned_clouds:
        out.append(P + shift)
    return out
def back_to_world(aligned_clouds, R_axis, axis_point):
    R_inv = R_axis.T
    clouds_w = []
    for P in aligned_clouds:
        Pw = (R_inv @ P.T).T + axis_point
        clouds_w.append(Pw)
    all_points = np.vstack(clouds_w)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(all_points)
    return clouds_w,merged_pcd

#####display clouds
def Plotter(clouds):
    plotter = pv.Plotter()
    if not isinstance(clouds, (list, tuple)):
        clouds = [clouds]
    for item in clouds:
        if isinstance(item, o3d.geometry.PointCloud):
            pts = np.asarray(item.points)
        else:
            pts = np.asarray(item)
        plotter.add_points(
            pv.PolyData(pts),
            point_size=3,
            render_points_as_spheres=True,
            lighting=True
        )
    plotter.show()
