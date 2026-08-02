import numpy as np
import cv2
import os
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from skimage import morphology
from model_phase import *

def show(Name,img):
    cv2.namedWindow(Name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(Name,img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def photo_process(imgs_path,phase_f):
    imgs = []  # 用来存放 8 张预处理后的图像

    for p in imgs_path:
        img = cv2.imread(p, 0)
        img = img.astype(np.float32)
        img = img / 255.0
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        # img_pro = bandpass_fringe(img,phase_f)
        imgs.append(img)
    return imgs

def Si_bu_bao_phase_safe(phase_imgs):

    I0 = cv2.imread(phase_imgs[0], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    I1 = cv2.imread(phase_imgs[1], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    I2 = cv2.imread(phase_imgs[2], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    I3 = cv2.imread(phase_imgs[3], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # 高斯滤波（注意需要将滤波结果赋值回去）
    kernel_size = (3, 3)
    sigma = 0.8
    I0 = cv2.GaussianBlur(I0, kernel_size, sigma)
    I1 = cv2.GaussianBlur(I1, kernel_size, sigma)
    I2 = cv2.GaussianBlur(I2, kernel_size, sigma)
    I3 = cv2.GaussianBlur(I3, kernel_size, sigma)
    numerator = I3 - I1
    denominator = I0 - I2
    # modulation = 0.5 * np.sqrt((I3 - I1) ** 2 + (I0 - I2) ** 2)
    # print("modulate:", modulation)
    # 避免除0问题（虽然 arctan2 本身能处理，但可能引发 warning）
    epsilon = 1e-6
    denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
    phi_wrapped = np.arctan2(numerator, denominator)  # 结果范围 [-π, π]
    phi_wrapped_cor=np.mod(phi_wrapped + 2 * np.pi, 2 * np.pi)  #限定范围为0~2pi 防止后续clip把负数削掉
    return phi_wrapped_cor

def Ba_bu_phase_safe(imgs):
    imgs_stack = np.stack(imgs, 0)
    sin_terms = np.sin(np.arange(8) * np.pi/4).astype(np.float32)
    cos_terms = np.cos(np.arange(8) * np.pi/4).astype(np.float32)
    numerator   = np.sum(imgs_stack * sin_terms[:, None, None], axis=0)
    denominator = np.sum(imgs_stack * cos_terms[:, None, None], axis=0)

    # --- modulation（关键步骤）---
    modulation = 0.25 * np.sqrt(numerator**2 + denominator**2)
    # 有效掩膜（避免大量 0 垃圾点）
    mask = modulation > 0.035     # 你可根据亮度调
    # 初始化为 nan（正确做法）
    phi = np.full_like(numerator, np.nan, dtype=np.float32)
    # 只在有效区域求相位
    phi[mask] = np.arctan2(numerator[mask], denominator[mask])
    # 映射到 0~2π
    phi = np.mod(phi + 2*np.pi, 2*np.pi)

    return phi

def show_phase(title,phi,cmap="hsv"):
    plt.figure(figsize=(6,5))
    plt.imshow(phi, cmap=cmap)
    plt.colorbar()
    # plt.title(title)
    plt.tight_layout()
    plt.show()

def show_histogram(data,name, bins=200):
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

if __name__ == '__main__':

    path_x = r'D:\PythonDoc\Structure_Light\opencv\crop_images\horizontal'
    path_y = r'D:\PythonDoc\Structure_Light\opencv\crop_images\vertical'
    images_x = os.listdir(path_x)   # 读入图像序列
    images_y = os.listdir(path_y)
    imgs_low_x = []  # 存储移相图片低频
    imgs_mid_x = []
    imgs_high_x = []
    imgs_low_y = []
    imgs_mid_y = []
    imgs_high_y = []
    # for f in sorted(glob.glob(os.path.join(path_gray, "*.png"))):
    #     img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    #     gray_pattern.append(img)
    for image in images_x:
        # print("读取到移相图片")
        img_path = os.path.join(path_x, image)
        if "f35" in str(image):
            imgs_low_x.append(img_path)
            phase_imgs_low_x = photo_process(imgs_low_x,35)
        elif "f36" in str(image):
            imgs_mid_x.append(img_path)
            phase_imgs_mid_x = photo_process(imgs_mid_x,36)
        elif "f37" in str(image):
            imgs_high_x.append(img_path)
            phase_imgs_high_x = photo_process(imgs_high_x,37)

    for image in images_y:
        # print("读取到移相图片")
        img_path = os.path.join(path_y, image)
        if "f35" in str(image):
            imgs_low_y.append(img_path)
            phase_imgs_low_y = photo_process(imgs_low_y, 35)
        elif "f36" in str(image):
            imgs_mid_y.append(img_path)
            phase_imgs_mid_y = photo_process(imgs_mid_y, 36)
        elif "f37" in str(image):
            imgs_high_y.append(img_path)
            phase_imgs_high_y = photo_process(imgs_high_y,37)
    # black_img = cv2.imread(black_path,cv2.IMREAD_GRAYSCALE)
    # white_img = cv2.imread(white_path, cv2.IMREAD_GRAYSCALE)
    # proj_grey_h, proj_grey_v, mask_grey = decode_graycode_full(gray_pattern,black_img,white_img,(854,480))
    # print("gray_u range:", np.nanmin(proj_grey_h), np.nanmax(proj_grey_h))
    # print("gray_v range:",np.nanmin(proj_grey_v), np.nanmax(proj_grey_v))
    # show_histogram(proj_grey_h,"proj_grey_h")
    # show_histogram(proj_grey_v, "proj_grey_v")
    # #
    phi_wrapped_low_x = Ba_bu_phase_safe(phase_imgs_low_x)
    phi_wrapped_mid_x = Ba_bu_phase_safe(phase_imgs_mid_x)
    phi_wrapped_heigh_x = Ba_bu_phase_safe(phase_imgs_high_x)
    phi_wrapped_low_y = Ba_bu_phase_safe(phase_imgs_low_y)
    phi_wrapped_mid_y = Ba_bu_phase_safe(phase_imgs_mid_y)
    phi_wrapped_heigh_y = Ba_bu_phase_safe(phase_imgs_high_y)
    deltas = np.linspace(0, 2 * np.pi, 8, endpoint=False) #num代表具体步法
    phi_low_abs_x, phi_mid_abs_x, phi_high_abs_x = Het_unwrap(phi_wrapped_low_x, phi_wrapped_mid_x, phi_wrapped_heigh_x,
                                                  35,36,37)
    phi_low_abs_y, phi_mid_abs_y, phi_high_abs_y = Het_unwrap(phi_wrapped_low_y, phi_wrapped_mid_y, phi_wrapped_heigh_y,
                                                  35,36,37)
    # show_phase("unwrap phase shift", phi_high_abs_x, cmap="gray")
    #质量评估

    phi_abs_clean_x = clean_noise(phi_high_abs_x)
    phi_abs_clean_y = clean_noise(phi_high_abs_y)

    # show("phi_high_abs_x:", phi_high_abs_x)
    # show("phi_high_abs_y:", phi_high_abs_y)
    # show_phase("clian_noise", phi_abs_clean_x, cmap="hsv")

    u_proj = phi_abs_clean_x * (854 / (2 * np.pi))  #将投影仪作为针孔模型
    v_proj = phi_abs_clean_y * (480 / (2 * np.pi))
    u_proj = np.clip(u_proj, 0, 854 - 1)
    v_proj = np.clip(v_proj, 0, 480 - 1)

    np.save("mid/u_pred.npy",u_proj)
    np.save("mid/v_pred.npy",v_proj)
    # #
    # show("u_proj",u_proj)
    # show("v_proj", v_proj)


    print("提取完成")




