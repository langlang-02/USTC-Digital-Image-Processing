import cv2
import numpy as np
import matplotlib.pyplot as plt

def sharpen_image(image):
    """高频增强（拉普拉斯锐化）"""
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def equalize_image(image):
    """直方图均衡化"""
    if len(image.shape) == 2:  # 灰度图
        return cv2.equalizeHist(image)
    else:  # 彩色图（对每个通道均衡化）
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# 读取图像（使用OpenCV自带的Lena图）
img = cv2.imread('lena.bmp', cv2.IMREAD_COLOR)  # 替换为你的Lena图片路径
if img is None:
    # 如果找不到图片，从网络下载（需要互联网连接）
    print("未找到本地lena.png，尝试从网络下载...")
    import urllib.request
    url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
    resp = urllib.request.urlopen(url)
    img = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)

# 转换为RGB格式用于显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 处理顺序1：先锐化后均衡化
sharp_first = sharpen_image(img)
result1 = equalize_image(sharp_first)

# 处理顺序2：先均衡化后锐化
equalized_first = equalize_image(img)
result2 = sharpen_image(equalized_first)

# 显示结果
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(sharp_first, cv2.COLOR_BGR2RGB))
plt.title('Sharpened First')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(result1, cv2.COLOR_BGR2RGB))
plt.title('Sharp → Equalize (Result)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
plt.title('Equalize → Sharp (Result)')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存结果
cv2.imwrite('result_sharp_first.png', result1)
cv2.imwrite('result_equalize_first.png', result2)