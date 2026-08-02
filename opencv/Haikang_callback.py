import ctypes
import os
import queue
import sys
import threading
import msvcrt
import numpy as np
import cv2
from ctypes import *

from matplotlib import pyplot as plt

from model_photo import *

sys.path.append(r"D:\MVS\Development\Samples\Python\MvImport")  # 换成自己的MVS中的路径
from MvCameraControl_class import *
from ctypes import *

CALLBACK_FUNC = WINFUNCTYPE(
    None,
    POINTER(c_ubyte),
    POINTER(MV_FRAME_OUT_INFO_EX),
    c_void_p
)

frame_queue = queue.Queue()  # 防止内存无限增长 最多20个
save_queue = queue.Queue()
g_bExit = False
# 自己在这个线程中修改，可以将相机获得的数据转换成opencv支持的格式，然后再用opencv操作

def image_callback(pData, pFrameInfo, pUser):
    frame_info = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
    nFrameLen = frame_info.nFrameLen
    img_data = np.frombuffer((c_ubyte * nFrameLen).from_address(addressof(pData.contents)), dtype=np.uint8)
    frame_queue.put(img_data, block=True)
    event.set()  # 通知consumer线
def thread_consumer():
    event.wait()
    while not g_bExit or not frame_queue.empty():
        try:
            img_data = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        # 处理
        temp = img_data.reshape((1080, 1440))  # 根据分辨率修改
        temp = cv2.cvtColor(temp, cv2.COLOR_BAYER_RG2BGR)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        # temp = cv2.xphoto.createSimpleWB().balanceWhite(temp)  #增加白平衡 避免相机仅捕获某一种颜色
        save_queue.put(temp)

def saver(direction,param=5):
    frame_count = 0
    while not g_bExit or not save_queue.empty():
        try:
            img_data = save_queue.get(timeout=1)
        except queue.Empty:
            continue
        if direction == 'full':
            filename = os.path.join(save_dir, f"frame_{frame_count + 1:04d}.png")
        # if "vertical" in str(direction):
        #     if "True" in str(direction):
        #         filename = os.path.join(path_V_T, f"vertical_T_{frame_count + 1:04d}.png")
        #     else:
        #         filename = os.path.join(path_V, f"vertical_{frame_count + 1:04d}.png")
        # if "horizontal" in str(direction):
        #     if "True" in str(direction):
        #         filename = os.path.join(path_H_T, f"horizontal_T_{frame_count + 1:04d}.png")
        #     else:
        #         filename = os.path.join(path_H, f"horizontal_{frame_count + 1:04d}.png")
        if "shift" in str(direction):
            if "horizontal" in str(direction):
                filename = os.path.join(path_horizontal, f"horizontal_f{param}_{frame_count + 1:04d}.png")
            if "vertical" in str(direction):
                filename = os.path.join(path_vertical, f"vertical_f{param}_{frame_count + 1:04d}.png")
            if "gray" in str(direction):
                filename = os.path.join(path_Gray, f"f_{frame_count + 1:04d}.png")
        if "calib" in str(direction):
            if "camera" in str(direction):
                filename = os.path.join(path_cam, f"camera_{frame_count + 1:04d}.png")
            else:
                for i in [1,2,3,4,5,6,7]:  # p1 到 p7
                    if f"p{i}" in str(param):
                        filename = os.path.join(path_projector, f"{param}_{frame_count + 1:04d}.png")
                        if True:
                            if frame_count == 0:
                                filename = os.path.join(path_projector, f"frame_black_{i}.png")
                            elif frame_count == 1:
                                filename = os.path.join(path_projector, f"frame_white_{i}.png")
                        break  # 找到就退出，不用继续检查
        cv2.imwrite(filename, img_data)
        print(f"Saved {filename}")
        frame_count += 1
def Camera_Init():
    # 获得设备信息
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    # nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            # 输出设备名字
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            # 输出设备ID
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        # 输出USB接口的信息
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
    # 选择设备
    nConnectionNum = input("please input the number of the device to connect:(default is 0):")
    if nConnectionNum == "":
        nConnectionNum = 0
    else:
        nConnectionNum = int(nConnectionNum)  # 如果输入了内容，则将其转换为整数

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()  #从这里正式调用相机

    # ch:选择设备并创建句柄 | en:Select device and create handle
    # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)
    ret = cam.MV_CC_StopGrabbing()
    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", 0)  # 1 开启 0关闭
    ret = cam.MV_CC_SetBoolValue("TriggerCacheEnable", True) #用于缓存信号，开启后，上个信号处理完毕前的信号会进行缓存，上个信号处理完成后，处理缓存的信号；不开启则同时接收的两个触发信号，仅当一个信号进行响应，默认不开启。
    ret = cam.MV_CC_SetEnumValue("LineSelector", 0) #用于对输入的信号进行硬件滤波用于选择输入源。0:Line0  ||2:Line2
    # ret = cam.MV_CC_SetIntValue("LineDebouncerTime", 0) ##用于设置滤波时间，单位为us
    ret = cam.MV_CC_SetEnumValue("ExposureMode", 0)  ## 0: Timed (固定曝光), 1: TriggerWidth (脉冲宽度曝光)
    # ret = cam.MV_CC_SetFloatValue("TriggerDelay_pre dark", 5000)  # 用于设定触发延时时间，接收到触发信号后，延时一段时间开始拍摄，默认为0，单位为us
    ret = cam.MV_CC_SetFloatValue("ExposureTime", 12000)  # 曝光时间 单位us
    ret = cam.MV_CC_SetEnumValue("TriggerSource", 0)  ##用于指定输入源，默认为Line0。
    ret = cam.MV_CC_SetEnumValue("TriggerActivation",0)  ##用于设定触发模式，默认为上升沿触发，0：Rising Edge  ||1：Falling Edge ||2：LevelHigh|| 3：LevelLow
    ret = cam.MV_CC_SetEnumValue("TriggerMode", 1)  # 1 开启 0关闭

    # 设置自动白平衡关闭
    ret = cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 0)  # 0: Off  1 once

    # #开启 gammer 校正
    ret = cam.MV_CC_SetBoolValue("GammaEnable", True)
    ret = cam.MV_CC_SetFloatValue("Gamma", 1.0) # 设置 Gamma 数值为为 1.0（线性）
    # 设置黑电平为 0
    ret = cam.MV_CC_SetIntValue("BlackLevel", 0)  #黑电平过小会
    # 导致红屏颜色异常，过大黑屏

    ret = cam.MV_CC_SetEnumValue("GainAuto", 0)  # 0: Off, 1: Once, 2: Continuous
    if ret != 0:
        print(f"关闭 GainAuto 失败，错误码: {ret}")
    else:
        print("已关闭 GainAuto")
    ret = cam.MV_CC_SetFloatValue("Gain", 0)

    return ret,cam

if __name__ == "__main__":
    #相机型号 MV—CA016—10UC
    save_dir = "images"
    save_exposion = "..\opencv\images\light"
    save_calib = r"images\calib"
    #处理相移图片
    path_horizontal = r"..\opencv\crop_images\horizontal"
    path_vertical = r"..\opencv\crop_images\vertical"
    path_Gray = r"..\opencv\crop_images\Gray_shift"
    #标定图片
    path_cam = os.path.join(save_calib,"camera")
    path_projector = os.path.join(save_calib, "p7")
    # # 创建文件夹
    for path in [path_Gray,path_horizontal,path_vertical,path_cam,path_projector]:
        os.makedirs(path, exist_ok=True)

    #清理标定文件夹
    clear_folder(os.path.join(save_calib, 'camera'))
    # clear_folder(os.path.join(save_calib, 'p1'))
    # clear_folder(os.path.join(save_calib, 'p2'))
    # clear_folder(os.path.join(save_calib, 'p3'))
    # clear_folder(os.path.join(save_calib, 'p4'))
    # clear_folder(os.path.join(save_calib, 'p5'))
    # clear_folder(os.path.join(save_calib, 'p6'))


    event = threading.Event()

    ret,cam = Camera_Init()
    # 从这开始，获取图片数据
    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    # csharp中没有memset函数，用什么代替？
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    # MV_CC_GetIntValue，获取Integer属性值，handle [IN] 设备句柄
    # strKey [IN] 属性键值，如获取宽度信息则为"Width"
    # pIntValue [IN][OUT] 返回给调用者有关相机属性结构体指针
    # 得到图片尺寸，这一句很关键
    # payloadsize，为流通道上的每个图像传输的最大字节数，相机的PayloadSize的典型值是(宽x高x像素大小)，此时图像没有附加任何额外信息

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()

    # Register callback
    # 转换为 C 回调
    c_image_callback = CALLBACK_FUNC(image_callback)
    ret = cam.MV_CC_RegisterImageCallBackEx(c_image_callback, None)
    if ret != 0:
        print("Register callback failed! ret[0x%x]" % ret)
        sys.exit()

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    consumer = threading.Thread(target=thread_consumer)  #注意修改参数
    saver = threading.Thread(target=saver,args=("calib_projector","p6",))

    consumer.start()
    saver.start()

    print("开始等待信号 Press any key to exit.")
    msvcrt.getch()
    g_bExit = True
    consumer.join()
    saver.join()

    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    print("Camera released. Program finished.")
