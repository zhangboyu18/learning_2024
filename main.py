import cv2
import numpy as np

video_path = 'D:\\road_video.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 创建背景消除器
# history:模型将考虑过去500帧的信息来构建背景
# varThreshold:用于控制背景模型中每个高斯分布的方差阈值 越小越容易被认为是前景
# detectShadows：会尝试检测并标记前景对象中的阴影部分
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
# fgbg = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=16, detectShadows=False)


while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 如果读取帧失败（例如视频结束），则退出循环
    if not ret:
        break

    # 应用背景消除器，返回一个与输入图像大小相同的二值图像（前景掩码）
    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask, np.ones((3, 3), np.uint8), iterations=2)
    fgmask = cv2.dilate(fgmask, np.ones((5, 5), np.uint8), iterations=3)

    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # 尝试通过颜色空间转换和阈值处理来减少阴影影响（这是一个简化的方法）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]  # 提取V通道（亮度）

    # 设定一个亮度阈值，低于该阈值的区域被认为是阴影候选区域
    threshold_value = 100  # 这个值需要根据实际情况调整
    _, shadow_mask = cv2.threshold(value_channel, threshold_value, 255, cv2.THRESH_BINARY_INV)
    fgmask = cv2.bitwise_and(fgmask, ~shadow_mask)

    # 将当前帧与前一帧进行或运算
    prev_foreground_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    if np.any(prev_foreground_mask) > 0:
        fgmask = cv2.bitwise_or(prev_foreground_mask, fgmask)
    prev_foreground_mask = fgmask.copy()

    # 查找前景掩码中的轮廓
    # cv2.RETR_EXTERNAL：轮廓检索模式，表示只检索最外层的轮廓。如果您想检索所有轮廓（包括嵌套的），可以使用cv.RETR_TREE或cv2RETR_LIST
    # cv2.CHAIN_APPROX_SIMPLE：Retrieval  轮廓逼近方法，它会压缩水平、垂直和对角线段，只保留它们的端
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓并在原始帧上绘制方框
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 忽略太小的轮廓（可能是噪声）
        if cv2.contourArea(contour) < 128 or w < 16 or h < 16:
            continue

        # 在原始帧上绘制方框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示处理后的帧
    cv2.imshow('Moving Object Detection1', fgmask)
    cv2.imshow('Moving Object Detection', frame)

    # # 按'q'键退出循环
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放视频捕获对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
