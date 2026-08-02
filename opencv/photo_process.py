
import os

"""
对 仿真 捕获图片进行裁剪减少不必要计算 保留待测物中心
"""
import numpy as np
import cv2
from model_photo import *

# # 图片在当前文件夹的位置
# file_in = 'capture_photos/horizontal'   # 原始图片存放位置
# file_out = 'desl_photoes/horizontal'   # 图片的保存位置

input_folde = r'images'
output_folder = r'crop_images'
process_subfolders(input_folde, output_folder)
# img_black = cv2.imread(r"images\frame_black_3.png",1)
# img_white = cv2.imread(r"images\frame_black_3.png",1)
# img_black_crop = process_image(img_black)
# img_white_crop = process_image(img_white)
# cv2.imwrite(os.path.join(output_folder,"img_black.png"),img_black_crop)
# cv2.imwrite(os.path.join(output_folder,"img_white.png"),img_white_crop)

