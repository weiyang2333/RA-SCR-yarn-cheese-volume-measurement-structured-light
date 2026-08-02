
from cloud_model import *

def Plotter(pcd,axis_dir = None,axis_point = None,axis_color="red",center_color="blue",
    center_radius=3.0,point_size=2,axis_length = 300,color=(0.68, 0.78, 0.90),background="white"):
    # plotter = pv.Plotter()
    plotter = pv.Plotter(off_screen=False)
    plotter.set_background(background)
    if isinstance(pcd, o3d.geometry.PointCloud):
        pts = np.asarray(pcd.points)
    else:
        pts = pcd
    # pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        print("点云为空")
        return
    poly = pv.PolyData(pts)
    # -------- 3. 显示旋转中心点 --------
    if axis_point is not None:
        axis_point = np.asarray(axis_point, dtype=float).reshape(3)
        center_sphere = pv.Sphere(radius=center_radius,center=axis_point)
        plotter.add_mesh(center_sphere, color=center_color, smooth_shading=True)
    # -------- 4. 显示旋转轴 --------
    if axis_point is not None and axis_dir is not None:
        axis_point = np.asarray(axis_point, dtype=float).reshape(3)
        axis_dir = np.asarray(axis_dir, dtype=float).reshape(3)
        norm = np.linalg.norm(axis_dir)
        if norm < 1e-12:
            print("axis_dir 长度过小，无法显示旋转轴")
        else:
            axis_dir = axis_dir / norm
            p1 = axis_point - axis_dir * (axis_length / 2.0)
            p2 = axis_point + axis_dir * (axis_length / 2.0)
            axis_line = pv.Line(p1, p2, resolution=1)
            plotter.add_mesh(axis_line, color=axis_color, line_width=4)

    # -------- 5. 可选：显示坐标轴 --------
    plotter.add_points(
        poly,
        color=color,
        point_size=point_size,
        render_points_as_spheres=True,
        lighting=True,
        ambient=0.30,
        diffuse=0.68,
        specular=0.25,
        specular_power=30
    )
    # plotter.add_points(
    #     pv.PolyData(pts),
    #     point_size=3,
    #     render_points_as_spheres=True,
    #     lighting=True)
    # plotter.add_points(
    #     poly,
    #     color=color,
    #     point_size=point_size,
    #     render_points_as_spheres=False,
    #     lighting=False,
    #     opacity=0.9
    # )

    # plotter.enable_eye_dome_lighting()
    # plotter.enable_anti_aliasing()  #抗锯齿
    plotter.show()


if __name__ == "__main__":
    yuanzhu = r"D:\PythonDoc\Structure_Light\cloud_rebuild\whole_clouds\ICP_cloud_cusha_duibi.ply"
    pcd_y = o3d.io.read_point_cloud(yuanzhu)
    center_small = np.load("center_small.npy")
    d = np.load(r"D:\PythonDoc\Structure_Light\cloud_rebuild\turntable_axis.npz")
    axis_dir = d["axis_dir"]
    Plotter(
        pcd_y,
        axis_dir=axis_dir,
        axis_point=center_small,
        axis_length=1400,
        center_radius=3.0
    )
