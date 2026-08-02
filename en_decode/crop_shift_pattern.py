import os

import cv2
import numpy as np
from model_photo import *

if __name__ == '__main__':
    folder_path = r'shift_patterns'
    output_folder_H = r'shift_H_8bit'
    output_folder_V = r'shift_V_8bit'
    output_folder_H = os.path.join("shift_patterns_8bit", output_folder_H) #这里 shift_patterns_8bit 是我建立的一个文件夹 存放两个新文件夹避免混乱
    output_folder_V = os.path.join("shift_patterns_8bit", output_folder_V)
    os.makedirs(output_folder_V, exist_ok=True)
    os.makedirs(output_folder_H, exist_ok=True)

    clear_folder(output_folder_H)
    clear_folder(output_folder_V)
    A = os.listdir(folder_path)
    for i in A:
        img_name = "{}".format(i)
        filename = os.path.splitext(img_name)[0]   #仅包含文件名字 去除后缀
        print("filnemam",filename)
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path,0)
        img_grey =  cv2.convertScaleAbs(img, alpha=(255.0/np.max(img)))
        # show("img",img)
        if ("vertical" in str(filename)):
            img_8bit = img_grey[0:1, :]
            output_path = os.path.join(output_folder_V, f"{filename}_8bit.png")
        if ("horizontal" in str(filename)):
            img_8bit = img_grey[:, 0:1]
            output_path = os.path.join(output_folder_H, f"{filename}_8bit.png")

        cv2.imwrite(output_path, img_8bit)


