import cv2
import numpy as np

# 读取灰度图像
image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    print("图像加载失败")
else:
     # 将前256行的像素值设为255
    image[:256, :] = 255

    # 打印处理后的图像（显示图像）
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('half0_lena.bmp', image)
