import os
import numpy as np
import cv2

"""
    生成单张 8 步相移正弦条纹图（uint8）

    参数：
      width, height : 图像尺寸
      freq          : 条纹频率（周期数），例如 35 表示沿该方向走 35 个周期
      step          : 相移步（1..8）
      direction     : "horizontal" 或 "vertical"
      I0            : 平均亮度（0..255）
      A             : 调制幅度（建议 0..127，且 I0±A 不要超出 0..255）
      phase0        : 额外初相（弧度）
    """
def make_8step_fringe(width, height, freq, step, direction, I0, A, phase0=0.0):
    if not (1 <= step <= 8):
        raise ValueError("step 必须在 1..8")
    if direction not in ("horizontal", "vertical"):
        raise ValueError('direction 只能是 "horizontal" 或 "vertical"')
    shift = (step - 1) * TWO_PI / 8 + float(phase0)
    #由于我自身购买的投影仪会翻转相位图因此需要将水平与竖直条纹提前翻转
    if direction == "horizontal":
        x = np.arange(width, dtype=np.float32) / float(width)
        phi = TWO_PI * float(freq) * x + shift
        phi = np.repeat(phi[None, :], height, axis=0)
    else:
        y = np.arange(height, dtype=np.float32) / float(height)
        phi = TWO_PI * float(freq) * y + shift
        phi = np.repeat(phi[:, None], width, axis=1)

    img = float(I0) + float(A) * np.cos(phi)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def generate_8step_structured_light_patterns(width,height,freqs,I0,A,save_dir,phase0=0.0,directions=("horizontal", "vertical"),):
    os.makedirs(save_dir, exist_ok=True)
    # 简单保护：避免全黑/全白饱和
    if  I0 + A > 255:
        raise ValueError(f"I0+A 超出 0..255：I0={I0}, A={A}，请调整（建议 I0=128, A<=127）")
    for freq in freqs:
        for step in range(1, 9):
            for direction in directions:
                img = make_8step_fringe(
                    width, height, freq, step, direction, I0, A, phase0=phase0
                )
                name = f"phase_{step}_f{freq}_{direction}.png"
                cv2.imwrite(os.path.join(save_dir, name), img)
                print("saved:", name)


if __name__ == "__main__":
    TWO_PI = 2 * np.pi
    width = 854   #根据你的投影仪分辨率调整
    height = 480
    freqs = (35, 36, 37)  # 所需要生成相位的频率
    I0 = 40  #平均亮度
    A = 200   #调制相位幅度
    save_dir_path = "shift"

    generate_8step_structured_light_patterns(
        width=width,
        height=height,
        freqs=freqs,
        I0=I0,
        A=A,
        save_dir=save_dir_path
    )