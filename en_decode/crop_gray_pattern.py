
import os
import shutil

import cv2
import numpy as np


def show(Name,img):
    cv2.namedWindow(Name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # phase_uint8 = img_norm.astype(np.uint8)
    cv2.imshow(Name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

if __name__ == '__main__':
    folder_path = r'pattern'
    output_folder_H = r"..\en_decode\pattern_1bit\H"
    output_folder_H_invert = r"..\en_decode\pattern_1bit\H_T"
    output_folder_V = r"..\en_decode\pattern_1bit\V"
    output_folder_V_invert = r"..\en_decode\pattern_1bit\V_T"
    # output_debruijn_H = r"..\en_decode\pattern_1bit\debruijn_H"
    # output_debruijn_V = r"..\en_decode\pattern_1bit\debruijn_V"
    clear_folder(output_folder_H)
    clear_folder(output_folder_H_invert)
    clear_folder(output_folder_V)
    clear_folder(output_folder_V_invert)
    # clear_folder(output_debruijn_H)
    # clear_folder(output_debruijn_V)

    A = os.listdir(folder_path)
    # print(A)
    for i in A:
        img_name = "{}".format(i)
        filename = os.path.splitext(img_name)[0]   #仅包含文件名字 去除后缀
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path,1)
        # print(img.shape)
        if( "horizontal" in str(filename)):
            if("True" in str(filename)):
                img_1bit = img[100:101, :]
                # show("img",img_1bit)
                output_path = os.path.join(output_folder_V_invert, f"{filename}_1bit.png")
            else:
                img_1bit = img[100:101, :]
                # show("img", img_1bit)
                output_path = os.path.join(output_folder_V, f"{filename}_1bit.png")
        if ("vertical" in str(filename)):
            if ("True" in str(filename)):
                img_1bit = img[:, 100:101]
                output_path = os.path.join(output_folder_H_invert, f"{filename}_1bit.png")
            else:
                img_1bit = img[:, 100:101]
                output_path = os.path.join(output_folder_H, f"{filename}_1bit.png")
        # if ("debruijn" in str(filename)):
        #     if ("horizontal" in str(filename)):
        #         img_1bit = img[0:1, :]
        #         output_path = os.path.join(output_debruijn_H, f"{filename}_1bit.png")
        #     else:
        #         img_1bit = img[:, 0:1]
        #         output_path = os.path.join(output_folder_H, f"{filename}_1bit.png")

        print("路径", output_path)
        print("当前路径",os.getcwd())
        cv2.imwrite(output_path, img_1bit)


    #制造一维纯白与纯黑图片
    img_black_h = np.zeros((480, 1), np.uint8)
    img_white_h = img_black_h.copy()
    img_white_h.fill(255)
    filename_white_h = os.path.join("..\en_decode\pattern_1bit\H", f"vertical_white.png")
    filename_black_h = os.path.join("..\en_decode\pattern_1bit\H", f"vertical_black.png")
    cv2.imwrite(filename_white_h,img_white_h)
    cv2.imwrite(filename_black_h,img_black_h)

    img_black_v = np.zeros((1, 854), np.uint8)
    img_white_v = img_black_v.copy()
    img_white_v.fill(255)
    filename_white_v = os.path.join("..\en_decode\pattern_1bit\V", f"horizontal_white.png")
    filename_black_v = os.path.join("..\en_decode\pattern_1bit\V", f"horizontal_black.png")
    cv2.imwrite(filename_white_v,img_white_v)
    cv2.imwrite(filename_black_v,img_black_v)
