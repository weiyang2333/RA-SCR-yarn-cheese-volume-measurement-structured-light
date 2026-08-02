from cloud_model import *
import copy
import os
import re
import matplotlib.pyplot as plt
import open3d as o3d
def fuse_angel_steps(
    clouds,          # [(name, pcd), ...]
    axis_point,
    axis_dir,
    angel_dir,
    voxel=3.0
):
    merged = o3d.geometry.PointCloud()
    aligned_clouds = []

    axis = np.asarray(axis_dir, dtype=float)
    axis /= np.linalg.norm(axis) + 1e-12
    axis_point = np.asarray(axis_point, dtype=float)

    for name, pcd in clouds:
        m = re.search(r'(\d+)', os.path.basename(name))
        idx = int(m.group(1))

        angle_deg = -angel_dir[idx]
        angle_rad = np.deg2rad(angle_deg)

        p = copy.deepcopy(pcd)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rad)
        p.rotate(R, center=axis_point)

        # o3d.visualization.draw_geometries([p, axis_line])
        aligned_clouds.append(p)   # 收集每一帧
        merged += p

    if voxel is not None:
        merged = merged.voxel_down_sample(voxel)
    merged, _ = merged.remove_statistical_outlier(
        nb_neighbors=30, std_ratio=2.0
    )

    return merged, aligned_clouds   #返回两个

def make_index_marker(center, idx, radius=2.5):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.translate(center)

    # 用 colormap 给不同 idx 不同颜色
    cmap = plt.get_cmap("tab20")
    color = cmap(idx % 20)[:3]
    sphere.paint_uniform_color(color)

    return sphere, color



if __name__ == "__main__":
    could_path = r"..\cloud_rebuild\axis_clouds\cu_tongzisha"
    could_file = ["cloud_0.ply", "cloud_1.ply", "cloud_2.ply", "cloud_2.ply", "cloud_4.ply", "cloud_5.ply",
                  "cloud_6.ply", "cloud_7.ply", "cloud_8.ply", "cloud_9.ply", "cloud_10.ply", "cloud_11.ply"]
    angel_dir = {0: 0, 1: 40.75, 2: 67.62, 3: 91.2, 4: 123.12, 5: 149.99, 6: 180.14, 7: 221.86,
                 8: 248.57, 9: 271.69, 10: 302.89, 11: 330.17}
    d = np.load(r"..\cloud_rebuild\turntable_axis.npz")
    axis_dir = d["axis_dir"]
    #这里需要先运行一下程序 注释掉下面这一行 之后生成 对应的center_small.npy文件
    center_small = np.load("center_small.npy")
    axis_line, point_sphere, line_dir = make_axis_line(axis_dir, center_small, length=400)
    axis_full = create_full_axes(length=300.0, origin=(0, 0, 0), line_width_hint=None)
    cloud_dir = []
    yuan_samll = []
    for i in could_file:
        file_path = os.path.join(could_path, i)
        pcd = o3d.io.read_point_cloud(file_path)
        pc_obj, _ = crop_pcd_xyz(pcd, y=(-100, 78))  # 对点云执行裁剪去掉圆台点云
        o3d.visualization.draw_geometries([pc_obj])
        clusters = split_pointcloud_by_distance(pcd, eps=3.0, min_samples=20,min_points=500)
        # 这个仅作展示作用
        # for j, c in enumerate(clusters):
        #     print(f"i:{i} ::  第c:{c}cluster  j:{j}: 点数 = {len(np.asarray(c.points))}")
        #     o3d.visualization.draw_geometries([clusters[j]])
        # yuan_samll.append(clusters[-2])
        pcd_crop = crop_pcd_local(pc_obj,axis_point=center_small,axis_dir=axis_dir,z=(-60, 50))
        cloud_dir.append([i,pcd_crop])
        # center_small, _, ratio_small = fit_axis_point_from_turntable_layer_ransac(yuan_samll[0], axis_dir)
        # print("center_small", center_small)
        # np.save("center_small.npy", center_small)
        # exit()

    # #点云旋转粗对齐
    merged, coarse_clouds = fuse_angel_steps(cloud_dir, axis_point=center_small, axis_dir=axis_dir, angel_dir=angel_dir)
    # o3d.visualization.draw_geometries(coarse_clouds)
   # 粗对齐结束 开始ICP微调
    aligned_z, opt = axis_constrained_icp_v2(coarse_clouds, axis_dir, center_small)
    aligned_z = move_back_axis(aligned_z, opt.cx, opt.cy)
    aligned_world, merged_pcd = back_to_world(
        aligned_z,R_axis=rotation_matrix_from_vectors(axis_dir, [0, 0, 1]),
        axis_point=center_small)

    # o3d.io.write_point_cloud("ICP_duibi_cloud_cusha.ply", merged_pcd)  #这里需要保存的话去掉注释 只是查看的话不用管
    # print("过渡点云保存完毕...")
    # #上色
    aligned_world_pcd = []
    N = len(aligned_world)
    for i, P in enumerate(aligned_world):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P)
        t = i / max(N - 1, 1)
        color = [
            0.0,  # R
            0.2 + 0.3 * t,  # G: 0.2 → 0.5
            0.6 + 0.4 * t  # B: 0.6 → 1.0
        ]
        pcd.paint_uniform_color(color)
        aligned_world_pcd.append(pcd)
    # visualize_with_index_markers(aligned_world_pcd, extra_geoms=[axis_line, point_sphere])

    # Plotter(aligned_world_pcd)


