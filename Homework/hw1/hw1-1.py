import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    print("图像加载失败")
else:
    # 定义区域的左上角坐标和大小
    x, y = 200, 200
    width, height = 10, 10

    # 读取指定区域的像素值
    region = image[y:y+height, x:x+width]

    # 打印像素值
    print("10x10 区域的像素值：")
    print(region)
