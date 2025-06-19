import cv2

def modify_bmp_palette(file_path):
    with open(file_path, 'rb+') as f:
        # BMP 文件头大小为 54 字节，调色板从第 54 字节开始
        f.seek(54)
        
        # 修改调色板，假设 BMP 是 8 位灰度图像
        for i in range(256):
            # 生成彩色映像表，例如: 蓝色渐变
            red = i
            green = 128
            blue = 255 - i
            f.write(bytes([blue, green, red, 0]))  # 写入 BGR0 格式

# 修改 BMP 文件
modify_bmp_palette('lena.bmp')

# 使用 OpenCV 打开并显示修改后的图像
image = cv2.imread('lena.bmp', cv2.IMREAD_UNCHANGED)
cv2.imshow('Modified Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('colored_lena.bmp', image)