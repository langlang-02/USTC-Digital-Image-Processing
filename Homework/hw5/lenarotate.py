import cv2
import numpy as np
import math

def rotate_image(image, angle, background_color=(0, 0, 0)):
    """
    实现图像任意角度旋转
    :param image: 输入图像(BGR格式)
    :param angle: 旋转角度(顺时针为正，单位：度)
    :param background_color: 填充背景颜色(BGR元组)
    :return: 旋转后的图像
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    
    # 计算旋转中心
    center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后图像的边界尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵的中心点偏移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 执行旋转并处理边界
    rotated = cv2.warpAffine(
        image, 
        M, 
        (new_w, new_h),
        borderValue=background_color
    )
    
    return rotated

# 读取图像(替换为你的lema.bmp路径)
image = cv2.imread('../Lena.bmp')
if image is None:
    raise FileNotFoundError("无法加载图像，请检查路径")

# 设置旋转角度(示例：30度)
angle = 30

# 执行旋转
rotated_image = rotate_image(image, angle, background_color=(255, 255, 255))

# 显示结果
cv2.imshow('Original', image)
cv2.imshow(f'Rotated {angle} degrees', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('rotated_lema.bmp', rotated_image)