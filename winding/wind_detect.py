
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import linregress
from sklearn.cluster import DBSCAN
#卷绕检测

def analyze_winding_structure(points, axis_dir):

    # ---------- 1 使用给定轴方向 ----------
    axis = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    axis = axis / np.linalg.norm(axis)

    # 参考点仍先取点云中心
    center = points.mean(axis=0)
    # ---------- 2 建立柱坐标 ----------
    v = axis
    diff = points - center
    h = diff @ v
    proj = np.outer(h, v) + center
    radial = points - proj
    r = np.linalg.norm(radial, axis=1)

    # ---------- 3 构造局部坐标 ----------
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(x_axis, v)) > 0.9:
        x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = np.cross(v, x_axis)
    u = u / np.linalg.norm(u)
    w = np.cross(v, u)
    w = w / np.linalg.norm(w)

    theta = np.arctan2(radial @ w, radial @ u)

    # ---------- 4 只保留外表面点 ----------
    r_threshold = np.percentile(r, 90)
    mask = r > r_threshold
    h_fit = h[mask]
    theta_fit = theta[mask]
    order = np.argsort(h_fit)
    h_fit = h_fit[order]
    theta_fit = theta_fit[order]
    # ---------- 5 半径 ----------
    # 取80%的外层点去除干扰
    radius = np.percentile(r, 80)
    # ---------- 6 DBSCAN寻找单条纱线 ----------
    # 展开角度  创新点部分
    theta_unwrap = np.unwrap(theta_fit)
    # 计算局部斜率
    slope = np.gradient(theta_unwrap, h_fit)
    # 去除异常值
    slope = slope[np.abs(slope) < 2]
    k = np.median(slope)
    print("k:", k)
    pitch = 2 * np.pi / np.abs(k)
    winding_density = 1 / pitch
    winding_angle = np.degrees(
        np.arctan((2 * np.pi * radius) / pitch)
    )
    direction = "right" if k > 0 else "left"

    result = {
        "旋转轴": axis,
        "筒子半径": radius,
        "螺距": pitch,
        "卷绕密度": winding_density,
        "卷绕角": winding_angle,
        "卷绕方向": "右卷绕" if direction == "right" else "左卷绕"
    }

    return result
#缺陷识别
def classify_winding_defects(points, window=2000):
    # PCA轴
    pca = PCA(n_components=3)
    pca.fit(points)
    axis = pca.components_[0]
    center = points.mean(axis=0)
    v = axis / np.linalg.norm(axis)
    diff = points - center
    h = diff @ v
    proj = np.outer(h, v) + center
    radial = points - proj
    r = np.linalg.norm(radial, axis=1)
    # 局部坐标
    x_axis = np.array([1,0,0])
    if abs(np.dot(x_axis, v)) > 0.9:
        x_axis = np.array([0,1,0])
    u = np.cross(v, x_axis)
    u = u / np.linalg.norm(u)
    w = np.cross(v, u)
    theta = np.arctan2(radial @ w, radial @ u)
    # 外层点
    r_threshold = np.percentile(r, 90)
    mask = r > r_threshold

    h = h[mask]
    theta = theta[mask]
    # 排序
    order = np.argsort(h)
    h = h[order]
    theta = theta[order]

    slopes = []

    for i in range(0, len(h) - window, window):

        h_win = h[i:i+window]
        t_win = theta[i:i+window]

        slope, _, _, _, _ = linregress(h_win, t_win)
        slopes.append(slope)

    slopes = np.array(slopes)

    std = slopes.std()
    mean = slopes.mean()

    # ---------- 分类 ----------
    # ---------- 缺陷分类 ----------
    if std < 0.05 * abs(mean):
        label = "正常卷绕"
    elif std < 0.15 * abs(mean):
        label = "叠丝"
    elif std < 0.3 * abs(mean):
        label = "塌陷"
    else:
        label = "乱卷"

    return {
        "平均卷绕斜率": mean,
        "卷绕斜率标准差": std,
        "卷绕状态": label
    }


if __name__ == "__main__":
    ply_path = "../cloud_rebuild/ICP_cloud.ply"
    d = np.load("../cloud_rebuild/turntable_axis.npz")
    axis_dir = d["axis_dir"]
    axis_point = d["axis_point"]
    cloud = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(cloud.points)
    result = analyze_winding_structure(points,axis_dir)
    print("卷绕结构")
    print(result)
    defect = classify_winding_defects(points)
    print("缺陷检测")
    print(defect)