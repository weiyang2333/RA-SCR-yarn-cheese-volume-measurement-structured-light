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


if __name__ == '__main__':
    images_x = []
    images_y = []
    path_x = f"../opencv/images/calib"
    path_y = f"../opencv/images/calib"
    for i in [1]:
        graycode_dir = f"../opencv/images/calib/p{i}"
        path_x = f"../opencv/images/calib/p{i}"
        path_y = f"../opencv/images/calib/p{i}"
        for f in sorted(glob.glob(os.path.join(graycode_dir, "*.png"))):
            filename = os.path.basename(f)
            if "horizontal" in filename:
                images_x.append(filename)
            if "vertical" in filename:
                images_y.append(filename)
    imgs_low_x = []  # 存储移相图片低频
    imgs_mid_x = []
    imgs_high_x = []
    imgs_low_y = []
    imgs_mid_y = []
    imgs_high_y = []
    for image in images_x:
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
    phi_abs_clean_x = clean_noise(phi_high_abs_x)
    phi_abs_clean_y = clean_noise(phi_high_abs_y)
    u_proj = phi_abs_clean_x * (854 / (2 * np.pi))  #将投影仪作为针孔模型
    v_proj = phi_abs_clean_y * (480 / (2 * np.pi))
    u_proj = np.clip(u_proj, 0, 854 - 1)
    v_proj = np.clip(v_proj, 0, 480 - 1)
    np.save(r"..\en_decode\mid\u_calib_pred.npy",phi_wrapped_heigh_x)
    np.save(r"..\en_decode\mid\v_calib_pred.npy",phi_wrapped_heigh_x)


    print("提取完成")




