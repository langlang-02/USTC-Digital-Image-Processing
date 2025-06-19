import numpy as np
import cv2
import math

def manual_rotate(image, angle_deg, background_color=(0, 0, 0)):
    """
    手动实现图像任意角度旋转
    :param image: 输入图像(BGR格式)
    :param angle_deg: 旋转角度(顺时针为正，单位：度)
    :param background_color: 填充背景颜色(BGR元组)
    :return: 旋转后的图像
    """
    # 转换为弧度
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 计算新图像尺寸
    new_w = int(abs(w * cos_a)) + int(abs(h * sin_a))
    new_h = int(abs(h * cos_a)) + int(abs(w * sin_a))
    
    # 创建新图像
    rotated = np.full((new_h, new_w, 3), background_color, dtype=np.uint8)
    
    # 计算中心点偏移
    cx = w / 2
    cy = h / 2
    new_cx = new_w / 2
    new_cy = new_h / 2
    
    # 逆向映射：遍历新图像的每个像素
    for y in range(new_h):
        for x in range(new_w):
            # 计算相对于新图像中心的坐标
            x_rel = x - new_cx
            y_rel = y - new_cy
            
            # 应用逆向旋转变换
            #计算原像素的理论位置
            src_x = x_rel * cos_a + y_rel * sin_a + cx
            src_y = -x_rel * sin_a + y_rel * cos_a + cy
            
            # 双线性插值
            if 0 <= src_x < w and 0 <= src_y < h:
                x1, y1 = int(src_x), int(src_y) #取整计算原像素理论位置最临近的4个像素位置
                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                
                # 计算权重
                dx = src_x - x1
                dy = src_y - y1
                
                # 四个相邻像素
                p11 = image[y1, x1]
                p21 = image[y1, x2]
                p12 = image[y2, x1]
                p22 = image[y2, x2]
                
                # 双线性插值计算
                rotated[y,x]= (
                    p11 * (1 - dx) * (1 - dy) +
                    p21 * dx * (1 - dy) +
                    p12 * (1 - dx) * dy +
                    p22 * dx * dy
                )
    
    return rotated

# 读取图像
image = cv2.imread('..\lena.bmp')
if image is None:
    raise FileNotFoundError("无法加载图像，请检查路径")

# 设置旋转角度(示例：30度)
angle = 30

# 执行旋转
rotated_image = manual_rotate(image, angle, background_color=(255, 255, 255))

# 显示结果
cv2.imshow('Original', image)
cv2.imshow(f'Rotated {angle} degrees (Manual)', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
# cv2.imwrite('rotated_lema_manual.bmp', rotated_image)