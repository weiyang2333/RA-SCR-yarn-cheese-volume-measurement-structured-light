import math
import shutil
from itertools import groupby
import cv2
import numpy as np
import os

def clear_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在！")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"无法删除 {item_path}，错误：{e}")

#逐像素线性分区 → 二进制 → Gray 转换 的方法，适合大区域、整屏投影。
def binary_to_gray(n):
    return n ^ (n >> 1)
def generate_graycode_broad(width, height, bit_num, direction, output_folder,
                      invert=False, step=2, gamma=2.2, border=6):

    os.makedirs(output_folder, exist_ok=True)

    # === [1] 计算最接近的整周期宽度 ===
    full_width = 2 ** bit_num             # 灰码完整周期长度
    repeats = int(np.ceil(width / full_width))  # 重复周期次数，确保能覆盖投影宽度
    actual_width = full_width * repeats         # 扩展后的总宽度
    start = (actual_width - width) // 2
    end = start + width

    print(f"[INFO] 方向={direction}, 目标宽度={width}, 灰码周期={full_width}, 实际生成宽度={actual_width}, 裁剪范围=({start}, {end})")

    # === [2] 生成灰码图案 ===
    for bit in range(bit_num):
        img = np.zeros((height, actual_width), dtype=np.uint8)

        # 横向灰码条纹生成（整周期）
        for x in range(0, actual_width, step):
            code_value = x // step
            gray_value = binary_to_gray(code_value % full_width)
            bit_value = (gray_value >> (bit_num - bit - 1)) & 1
            if invert:
                bit_value = 1 - bit_value
            img[:, x:x + step] = bit_value * 255

        # 裁剪到实际投影宽度
        img = img[:, start:end]

        # === [3] 垂直方向支持 ===
        if direction == 'vertical':
            img = img.T  # 注意：转置后分辨率变为 (width, height)

        # === [4] Gamma 校正 ===
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, lut)

        # === [5] 添加边框保护 ===
        if border > 0:
            cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), 0, border)

        # === [6] 保存 ===
        filename = os.path.join(output_folder, f"{direction}_{bit}_{invert}.png")
        cv2.imwrite(filename, img)
        print(f"✅ 已保存: {filename} (step={step}, gamma={gamma}, border={border})")


def generate_graycode(width, height, bit_num, direction, output_folder, invert=False):
    os.makedirs(output_folder, exist_ok=True)
    full_width = 2 ** bit_num  # 生成最近的 2^n 宽度 (>= 投影仪宽度)

    start = (full_width - width) // 2
    end = start + width

    for bit in range(bit_num):
        img = np.zeros((height, full_width), dtype=np.uint8)

        for x in range(full_width):
            code_value = x
            gray_value = binary_to_gray(code_value)
            bit_value = (gray_value >> (bit_num - bit - 1)) & 1
            if invert:
                bit_value = 1 - bit_value
            img[:, x] = bit_value * 255

        # 裁剪到投影仪分辨率
        img = img[:, start:end]

        if direction == 'vertical':
            img = img.T

        filename = os.path.join(output_folder, f"{direction}_{bit}_{invert}.png")
        cv2.imwrite(filename, img)
        print(f"已保存: {filename} (裁剪范围: {start}-{end})")

#直接“分块式”Gray 码

def generate_blockwise_graycode(width, height, bit_num, direction, output_folder,
                                block_size=4, invert=False):
    """
    生成真正分块独立 GrayCode
    - 每个水平/竖直块内部独立 GrayCode
    - direction: 'horizontal' 或 'vertical'
    - block_size: 水平方向块数（horizontal）或竖直方向块数（vertical）
    """
    os.makedirs(output_folder, exist_ok=True)

    if direction == 'horizontal':
        block_w = width // block_size
        for bit in range(bit_num):
            img = np.zeros((height, width), dtype=np.uint8)
            for bx in range(block_size):
                x0 = bx * block_w
                x1 = width if bx == block_size-1 else x0 + block_w
                # 块内部独立 GrayCode
                for i in range(block_w):
                    idx = int(i / block_w * (2**bit_num))
                    g = binary_to_gray(idx)
                    bit_val = (g >> (bit_num - bit - 1)) & 1
                    if invert:
                        bit_val ^= 1
                    img[:, x0 + i] = bit_val * 255
            filename = os.path.join(output_folder,
                        f"blockwise_{direction}_bit{bit}_{'inv' if invert else 'nor'}.png")
            cv2.imwrite(filename, img)

    elif direction == 'vertical':
        block_h = height // block_size
        for bit in range(bit_num):
            img = np.zeros((height, width), dtype=np.uint8)
            for by in range(block_size):
                y0 = by * block_h
                y1 = height if by == block_size-1 else y0 + block_h
                # 块内部独立 GrayCode
                for j in range(block_h):
                    idx = int(j / block_h * (2**bit_num))
                    g = binary_to_gray(idx)
                    bit_val = (g >> (bit_num - bit - 1)) & 1
                    if invert:
                        bit_val ^= 1
                    img[y0 + j, :] = bit_val * 255
            filename = os.path.join(output_folder,
                        f"blockwise_{direction}_bit{bit}_{'inv' if invert else 'nor'}.png")
            cv2.imwrite(filename, img)
    print(f"生成完成: {direction}, bit_num={bit_num}, block_size={block_size}, invert={invert}")

# 参数
width = 854
height = 480
path = "pattern"
clear_folder(path)
# 生成 X 方向 10 bit
generate_graycode(width, height, bit_num=10, direction='horizontal', output_folder="pattern")
#生成X方向反向格雷码
generate_graycode(width, height, bit_num=10, direction='horizontal', output_folder="pattern",invert = True)

# 生成 Y 方向 9 bit
generate_graycode(height, width, bit_num=9, direction='vertical', output_folder="pattern")
# 生成 Y 反方向 9 bit
generate_graycode(height, width, bit_num=9, direction='vertical', output_folder="pattern",invert = True)




