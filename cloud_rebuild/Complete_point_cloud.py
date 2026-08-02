from cloud_model import *
import copy
import numpy as np
import open3d as o3d

def _safe_normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("向量长度过小，无法归一化")
    return v / n


def _rotation_matrix_from_vectors(src, dst, eps=1e-12):

    a = _safe_normalize(src)
    b = _safe_normalize(dst)

    v = np.cross(a, b)
    c = np.dot(a, b)

    # 同向
    if np.linalg.norm(v) < eps and c > 0:
        return np.eye(3, dtype=np.float64)

    # 反向：需要找一个与 a 不平行的轴旋转 180°
    if np.linalg.norm(v) < eps and c < 0:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis = np.cross(a, tmp)
        axis = _safe_normalize(axis)

        # Rodrigues，角度 pi
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float64)
        R = np.eye(3) + 2.0 * (K @ K)  # sin(pi)=0, 1-cos(pi)=2
        return R

    s = np.linalg.norm(v)
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=np.float64)

    R = np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2))
    return R


def estimate_cloud_main_axis(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("点云点数太少，无法估计主轴")

    center = pts.mean(axis=0)
    X = pts - center

    # SVD 最大奇异值对应主方向
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    axis = vh[0]
    axis = _safe_normalize(axis)

    return axis

def estimate_top_plane_normal(pcd: o3d.geometry.PointCloud,ratio=0.6,ref_dir=None,):
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] < 10:
        raise ValueError("点云点数太少")

    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio 必须在 (0, 1] 范围内")

    # 1. 整体中心
    center0 = pts.mean(axis=0)

    # 2. 计算到中心距离
    rr = np.linalg.norm(pts - center0, axis=1)
    r_max = np.max(rr)

    if r_max < 1e-12:
        raise ValueError("点云半径过小")

    # 3. 按比例取圆形区域
    r_keep = r_max * ratio
    idx = np.where(rr <= r_keep)[0]

    if len(idx) < 3:
        raise ValueError("按当前 ratio 选出的点太少，无法拟合平面")

    sub_pts = pts[idx]
    center = sub_pts.mean(axis=0)

    # 4. PCA / SVD 拟合平面法向
    X = sub_pts - center
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    # 5. 用参考方向统一正负
    if ref_dir is not None:
        ref_dir = np.asarray(ref_dir, dtype=np.float64).reshape(3)
        ref_dir = ref_dir / np.linalg.norm(ref_dir)
        if np.dot(normal, ref_dir) < 0:
            normal = -normal

    return normal, idx, center
def rotate_pointcloud_axis_to_horizontal(
    pcd: o3d.geometry.PointCloud,
    target_axis=(1, 0, 0),
    source_axis=None,
    use_pca_if_source_none=True,
    rotate_center="center",
    return_transform=False,
):

    if source_axis is None:
        source_axis = estimate_cloud_main_axis(pcd)
    source_axis = _safe_normalize(source_axis)
    target_axis = _safe_normalize(target_axis)

    R = _rotation_matrix_from_vectors(source_axis, target_axis)
    pcd_rot = copy.deepcopy(pcd)

    if rotate_center == "center":
        center = pcd_rot.get_center()
    elif rotate_center == "origin":
        center = np.zeros(3, dtype=np.float64)
    else:
        center = np.asarray(rotate_center, dtype=np.float64).reshape(3)

    pcd_rot.rotate(R, center=center)

    if return_transform:
        return pcd_rot, R, center
    return pcd_rot


def translate_cloud_horizontally_by_center(
    pcd: o3d.geometry.PointCloud,
    fitted_center_3d,
    axis_dir,
    target_uv=(0.0, 0.0),
):
    """
    根据拟合圆心，只做水平平移
    不改变 axis_dir 方向上的轴向位置
    """
    axis_dir = np.asarray(axis_dir, dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    u, v = make_uv(axis_dir)

    fitted_center_3d = np.asarray(fitted_center_3d, dtype=np.float64).reshape(3)
    target_uv = np.asarray(target_uv, dtype=np.float64).reshape(2)

    cur_u = fitted_center_3d @ u
    cur_v = fitted_center_3d @ v

    du = target_uv[0] - cur_u
    dv = target_uv[1] - cur_v

    trans = du * u + dv * v

    out = copy.deepcopy(pcd)
    out.translate(trans)
    return out, trans
def crop_top_cloud_outer_ring(
    pcd: o3d.geometry.PointCloud,
    center_3d,
    axis_dir,
    keep_radius,
):
    pts = np.asarray(pcd.points, dtype=np.float64)
    center_3d = np.asarray(center_3d, dtype=np.float64).reshape(3)
    axis_dir = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    vec = pts - center_3d
    t = vec @ axis_dir
    proj = np.outer(t, axis_dir)
    radial = vec - proj
    r = np.linalg.norm(radial, axis=1)

    mask = r <= keep_radius

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])

    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[mask])

    return out, mask
def move_top_cloud_center_to_side_top(
    pcd_top: o3d.geometry.PointCloud,
    top_center_3d,
    side_cloud: o3d.geometry.PointCloud,
    axis_point,
    axis_dir,
    top_quantile=0.995,
    extra_offset=0.0,
):
    """
    将 top 点云按照其中心点 top_center_3d，
    平移到 side_cloud 顶部的旋转轴位置

    参数
    ----
    pcd_top : 顶部点云
    top_center_3d : 顶部点云拟合得到的红点中心
    side_cloud : 侧面融合后的点云
    axis_point, axis_dir : 侧面点云旋转轴
    top_quantile : 顶部位置分位数，避免极个别毛刺点
    extra_offset : 额外轴向偏移，单位与点云一致
                   >0 表示沿轴正方向再抬高一点
                   <0 表示向下压一点
    """
    axis_point = np.asarray(axis_point, dtype=np.float64).reshape(3)
    axis_dir = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    top_center_3d = np.asarray(top_center_3d, dtype=np.float64).reshape(3)

    pts_side = np.asarray(side_cloud.points, dtype=np.float64)
    if pts_side.shape[0] < 10:
        raise ValueError("side_cloud 点数太少")

    t_side = (pts_side - axis_point) @ axis_dir
    t_top = np.quantile(t_side, top_quantile) + extra_offset

    target_center = axis_point + t_top * axis_dir
    trans = target_center - top_center_3d

    pcd_top_moved = copy.deepcopy(pcd_top)
    pcd_top_moved.translate(trans)

    return pcd_top_moved, trans, target_center

#####对于侧面与顶部抽取接触范围并融合
def fast_force_icp_fuse(
    source_pcd,target_pcd,force_dist=100.0,refine_dist=8.0,
    voxel_size=2.0,icp_iter=60,merge_voxel=0.8,force_axis_dir=None,
    sample_step=5,
):
    src_full = copy.deepcopy(source_pcd)
    tgt_full = copy.deepcopy(target_pcd)

    # 1. 降采样，用于计算粗平移和 ICP
    src = src_full.voxel_down_sample(voxel_size)
    tgt = tgt_full.voxel_down_sample(voxel_size)

    src_pts = np.asarray(src.points, dtype=np.float64)
    tgt_pts = np.asarray(tgt.points, dtype=np.float64)

    if len(src_pts) == 0 or len(tgt_pts) == 0:
        raise ValueError("source 或 target 点云为空")

    # 2. KDTree 找 source 到 target 最近邻
    tgt_tree = o3d.geometry.KDTreeFlann(tgt)

    trans_list = []
    dist_list = []

    for p in src_pts[::sample_step]:
        k, idx, dist2 = tgt_tree.search_knn_vector_3d(p, 1)
        if k <= 0:
            continue

        d = np.sqrt(dist2[0])

        # 只使用距离小于 force_dist 的对应点
        if d <= force_dist:
            q = tgt_pts[idx[0]]
            trans_list.append(q - p)
            dist_list.append(d)

    if len(trans_list) < 30:
        print("[WARN] 可用于强制拉近的对应点太少，跳过粗平移")
        coarse_trans = np.zeros(3)
    else:
        trans_arr = np.asarray(trans_list)
        dist_arr = np.asarray(dist_list)

        # 3. 去掉最差的 20% 对应，避免飞点影响
        keep_th = np.quantile(dist_arr, 0.80)
        good = dist_arr <= keep_th
        trans_arr = trans_arr[good]

        # 4. 用中位数平移，抗异常点
        coarse_trans = np.median(trans_arr, axis=0)

        # 如果给了旋转轴，只沿轴向拉近
        if force_axis_dir is not None:
            axis = np.asarray(force_axis_dir, dtype=np.float64).reshape(3)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            coarse_trans = np.dot(coarse_trans, axis) * axis

    print("[force ICP] coarse_trans =", coarse_trans)
    print("[force ICP] used correspondences =", len(trans_list))

    # 5. 把粗平移应用到完整 source 点云
    src_full.translate(coarse_trans)

    # 6. 再做正常 ICP 精修
    src_icp = src_full.voxel_down_sample(voxel_size)
    tgt_icp = tgt_full.voxel_down_sample(voxel_size)

    result = o3d.pipelines.registration.registration_icp(
        src_icp,
        tgt_icp,
        max_correspondence_distance=refine_dist,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=icp_iter
        )
    )

    # 7. 应用 ICP 变换到完整 source
    source_aligned = copy.deepcopy(src_full)
    source_aligned.transform(result.transformation)

    # 8. 合并
    merged = copy.deepcopy(target_pcd)
    merged += source_aligned

    if merge_voxel is not None and merge_voxel > 0:
        merged = merged.voxel_down_sample(merge_voxel)

    print("[force ICP] ICP fitness:", result.fitness)
    print("[force ICP] ICP inlier_rmse:", result.inlier_rmse)
    print("[force ICP] ICP transformation:\n", result.transformation)

    return merged, source_aligned, result, coarse_trans

def Plotter_industrial(clouds, point_size=4):
    if not isinstance(clouds, (list, tuple)):
        clouds = [clouds]
    all_pts = []
    for item in clouds:
        if isinstance(item, o3d.geometry.PointCloud):
            pts = np.asarray(item.points)
        else:
            pts = np.asarray(item)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            continue

        all_pts.append(pts)
    if len(all_pts) == 0:
        print("点云为空，无法显示")
        return

    all_pts_np = np.vstack(all_pts)
    center = all_pts_np.mean(axis=0)
    extent = np.ptp(all_pts_np, axis=0)
    size = np.max(extent)
    print("点云数量:", len(all_pts_np))
    print("点云中心:", center)
    print("点云范围:", extent)
    plotter = pv.Plotter(window_size=(900, 700))
    plotter.set_background((0.94, 0.94, 0.94))
    for pts in all_pts:
        poly = pv.PolyData(pts)
        plotter.add_points(
            poly,
            color=(0.60, 0.60, 0.60),
            point_size=point_size,
            render_points_as_spheres=True,
            lighting=True,
            ambient=0.35,
            diffuse=0.75,
            specular=0.15,
            specular_power=25,
        )
    plotter.enable_eye_dome_lighting()
    plotter.enable_parallel_projection()
    plotter.hide_axes()
    # 关键：相机看向点云中心，而不是看向原点
    plotter.camera_position = [
        center + np.array([1.8 * size, -2.4 * size, 1.4 * size]),
        center,
        (0, 0, 1),]
    plotter.reset_camera()
    plotter.show()

if __name__ == "__main__":
    bottom = r"..\cloud_rebuild\axis_clouds\duibi_tongzisha\cloud_bottom.ply"
    top = r"..\cloud_rebuild\axis_clouds\duibi_tongzisha\cloud_top.ply"
    bottom_save = r"..\cloud_rebuild\san_cloud\cloud_duibi_bottom.ply"
    top_save = r"..\cloud_rebuild\san_cloud\cloud_duibi_top.ply"
    icp_cloud = r"..\cloud_rebuild\ICP_duibi_cloud_cusha.ply"

    pcd_top_chu = o3d.io.read_point_cloud(top)
    pcd_bottom_chu = o3d.io.read_point_cloud(bottom)
    icp_cloud = o3d.io.read_point_cloud(icp_cloud)

    d = np.load(r"..\cloud_rebuild\turntable_axis.npz")
    axis_dir = d["axis_dir"]
    axis_point = np.load("center_small.npy")
    axis_line, point_sphere, line_dir = make_axis_line(axis_dir, axis_point, length=400)
    clusters_top = split_pointcloud_by_distance(pcd_top_chu, eps=3.0, min_samples=20, min_points=500)
    clusters_bottom = split_pointcloud_by_distance(pcd_bottom_chu, eps=3.0, min_samples=20, min_points=500)
    # o3d.visualization.draw_geometries([clusters[0]])
    pcd_top = clusters_top[0]
    pcd_bottom = clusters_bottom[0]
    icp_cloud = crop_pcd_local(icp_cloud,axis_point=axis_point,axis_dir=axis_dir,y=(-60, 180))

    top_normal, inliers, plane_center = estimate_top_plane_normal(pcd_top,ratio=0.55,ref_dir=axis_dir)
    bottom_normal, inliers, plane_center = estimate_top_plane_normal(pcd_bottom, ratio=0.55, ref_dir=axis_dir)
    # 保证法向方向和侧面旋转轴尽量同向
    if np.dot(top_normal, axis_dir) < 0:
        top_normal = top_normal      #这里正负号决定你的点云朝向
    if np.dot(bottom_normal, axis_dir) < 0:
        bottom_normal = bottom_normal      #这里正负号决定你的点云朝向

    pcd_top_r = rotate_pointcloud_axis_to_horizontal(
        pcd_top,
        source_axis=top_normal,  # 当前竖直方向
        target_axis=-axis_dir,  # 目标水平方向
        rotate_center="center"
    )
    pcd_bottom_r = rotate_pointcloud_axis_to_horizontal(
        pcd_bottom,
        source_axis=bottom_normal,  # 当前竖直方向
        target_axis=-axis_dir,  # 目标水平方向
        rotate_center="center"
    )
    # o3d.visualization.draw_geometries([pcd_top_b,axis_line,point_sphere])
    # exit()

    top_center = pcd_top.get_center()
    bottom_center = pcd_bottom.get_center()

    # 生成一个红色小球表示中心点
    center_ball = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
    center_ball.translate(top_center)
    center_ball.paint_uniform_color([1, 0, 0])
    center_ball.compute_vertex_normals()
    # o3d.visualization.draw_geometries([pcd_top_h,center_ball])
    #
    # print("拟合圆心:", fit_res["center_3d"])
    # print("拟合半径:", fit_res["radius"])
    # print("内点比例:", fit_res["inlier_ratio"])
    #
    pcd_top_crop, mask = crop_top_cloud_outer_ring(
        pcd_top_r,
        center_3d=top_center,  # 或 fit_res["center_3d"] 平移后的中心
        axis_dir=axis_dir,
        keep_radius=80,  # 这里自己调裁剪圆形点云的大小 去除边缘不稳定的噪声
    )
    pcd_bottom_crop, mask = crop_top_cloud_outer_ring(
        pcd_bottom_r,
        center_3d=bottom_center,  # 或 fit_res["center_3d"] 平移后的中心
        axis_dir=axis_dir,
        keep_radius=80,  # 这里自己调裁剪圆形点云的大小 去除边缘不稳定的噪声
    )
    # o3d.visualization.draw_geometries([pcd_top_crop])
    #
    pcd_top_moved, trans, target_center = move_top_cloud_center_to_side_top(
        pcd_top=pcd_top_crop,  # 或者你真正准备拼接的 top 点云
        top_center_3d=top_center,
        side_cloud=icp_cloud,
        axis_point=axis_point,
        axis_dir=axis_dir,
        top_quantile=0.97,
        extra_offset=0.0)
    pcd_bottom_moved, trans, target_center = move_top_cloud_center_to_side_top(
        pcd_top=pcd_bottom_crop,  # 或者你真正准备拼接的 top 点云
        top_center_3d=bottom_center,
        side_cloud=icp_cloud,
        axis_point=axis_point,
        axis_dir=axis_dir,
        top_quantile=0.97,
        extra_offset=-145.5)
    # # print("水平平移向量:", trans)
    # o3d.visualization.draw_geometries([pcd_top_moved,icp_cloud,axis_line,point_sphere])
    # Plotter([pcd_top_moved,pcd_bottom_moved,icp_cloud])
    # Plotter([ pcd_top_moved,pcd_bottom_moved, icp_cloud])

    # 1. 顶部融合到侧面
    merged_top, top_icp, result_top, coarse_trans_top = fast_force_icp_fuse(
        source_pcd=pcd_top_moved,
        target_pcd=icp_cloud,
        force_dist=100.0,  # 这里允许 100 mm 内的点参与粗拉近
        refine_dist=8.0,  # 真正 ICP 精配准只用 8 mm
        voxel_size=2.0,
        icp_iter=60,
        merge_voxel=0.8,
        force_axis_dir=axis_dir,  # 建议先加这个，只沿旋转轴方向强行贴近
    )
    Plotter([ merged_top])

    # 2. 底部融合到已经融合顶部后的点云
    merged_all, bottom_icp, result_bottom, coarse_trans_bottom = fast_force_icp_fuse(
        source_pcd=pcd_bottom_moved,
        target_pcd=merged_top,
        force_dist=100.0,
        refine_dist=8.0,
        voxel_size=2.0,
        icp_iter=60,
        merge_voxel=0.8,
        force_axis_dir=axis_dir,
    )
    merged = o3d.io.read_point_cloud(r"whole_clouds/ICP_cloud_cusha_duibi.ply")
    # Plotter_industrial([merged_all], point_size=4)
    # o3d.io.write_point_cloud(r"whole_clouds/ICP_cloud_cusha_duibi.ply", merged_all)
    icp_cloud = o3d.io.read_point_cloud("whole_clouds/ICP_cloud_cusha_duibi.ply")
    o3d.visualization.draw_geometries([icp_cloud])




