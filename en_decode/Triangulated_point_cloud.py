import os
import sys
import numpy as np
import cv2
import open3d as o3d
from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# === 1. 读取标定数据 ===
data = np.load(r"..\opencv\calibration_params.npz")
K_cam, D_cam = data["K_cam"], data["D_cam"]
K_proj, D_proj = data["K_proj"], data["D_proj"]
R, T = data["R"], data["T"].reshape(3,) #注意 这里的RT是相机到投影仪 我们后续也是以相机视野建立坐标系

# === 2. 读取 u_pred / v_pred ===
u_pred = np.load(r"mid\u_pred.npy").astype(np.float64)
v_pred = np.load(r"mid\v_pred.npy").astype(np.float64)
print("u_pred",np.nanmin(u_pred), np.nanmax(u_pred))
print("v_pred",np.nanmin(v_pred), np.nanmax(v_pred))
# === 3. 选取有效像素 ===
mask = (u_pred > 0) & (v_pred > 0) & np.isfinite(u_pred) & np.isfinite(v_pred)
ys, xs = np.nonzero(mask)
u_proj = u_pred[ys, xs]
v_proj = v_pred[ys, xs]

# === 4. 去畸变（相机端 & 投影仪端） ===
pts_cam = np.stack([xs, ys], axis=1).astype(np.float64).reshape(-1,1,2)
pts_proj = np.stack([u_proj, v_proj], axis=1).astype(np.float64).reshape(-1,1,2)
print(f"[INFO] 匹配成功点数: {len(pts_cam)}")

pts_cam_norm = cv2.undistortPoints(pts_cam.reshape(-1,1,2), K_cam, D_cam)
pts_proj_norm = cv2.undistortPoints(pts_proj.reshape(-1,1,2), K_proj, D_proj)
# === 5. 三角化 ===   说明一下 点云三角化计算有两种方式 理论上畸变矫正好的话 结果是近似一样的
# P1 =  K_cam @ np.hstack([np.eye(3), np.zeros((3,1))]).astype(np.float64)
# P2 =  K_proj @ np.hstack([R, T.reshape(3,1)]).astype(np.float64)

P1 =  np.hstack([np.eye(3), np.zeros((3,1))]).astype(np.float64)
P2 =  np.hstack([R, T.reshape(3,1)]).astype(np.float64)

pts4d = cv2.triangulatePoints(P1, P2,
    pts_cam_norm.reshape(-1,2).T,
    pts_proj_norm.reshape(-1,2).T)
# pts4d = cv2.triangulatePoints(P1, P2,
#                               pts_cam.reshape(-1,2).T,
#                               pts_proj.reshape(-1,2).T)
pts3d = (pts4d[:3] / pts4d[3]).T
pts3d = pts3d[np.isfinite(pts3d).all(axis=1)]
mask_z = np.isfinite(pts3d[:,2]) & (pts3d[:,2] > 0) & (pts3d[:,2] < 3000)
print(f"[INFO] 三角化输出点数: {len(pts3d)}")
print(f"[INFO] 有效深度点数: {np.sum(mask_z)}")
print(f"[INFO] 有效率: {np.sum(mask_z)/len(xs)*100:.2f}%")

# === 7. 输出点云 ===

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts3d)

# 上色（深度伪彩）
# z = pts3d[:, 0]
# z_min, z_max = np.percentile(z, [2, 98])
# colors = (z - z_min) / (z_max - z_min + 1e-8)
# colors = np.clip(colors, 0, 1)
# pcd.colors = o3d.utility.Vector3dVector(np.stack([colors, 1 - colors, np.zeros_like(colors)], axis=1))


# 降采样 + 去噪
# pcd1, _ = pcd.remove_statistical_outlier(nb_neighbors=4, std_ratio=2.2)
pcd_clean, ind = pcd.remove_radius_outlier(nb_points=20, radius=5)## 统计滤波：nb_neighbors 越大越平滑，std_ratio 越小越严格
# pcd_clean, ind = pcd.remove_radius_outlier(nb_points=10, radius=2)## 筒m子纱
#把点云按 0.5 mm（或 0.5 单位）大小划分成立方体体素，每个体素只保留一个点，从而减少点数量。
# pcd_clean = pcd_clean.voxel_down_sample(voxel_size=0.5)
# pcd_clean.points = o3d.utility.Vector3dVector(pts3d * np.array([1, 1, 1]))

#显示点云
o3d.visualization.draw_geometries([pcd_clean])

#保存点云  这个 if 是针对快速脚本保存用的，如果单次调试，下面的保存路径自行调整
if len(sys.argv) > 1:
    cloud_idx = sys.argv[1].strip()
else:
    print("未检测到点云编号输入，请重新输入！")
o3d.io.write_point_cloud(f"../cloud_rebuild/axis_clouds/duibi_tongzisha/cloud_{cloud_idx}.ply", pcd_clean)
# o3d.io.write_point_cloud("../cloud_rebuild/clouds/cloud_11.ply", pcd_clean)

