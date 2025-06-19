import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalize_image(image):
    """直方图均衡化"""
    if len(image.shape) == 2:  # 灰度图
        return cv2.equalizeHist(image)
    else:  # 彩色图（对Y通道均衡化）
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def plot_histogram(image, ax, title):
    """绘制直方图"""
    if len(image.shape) == 2:  # 灰度图
        ax.hist(image.ravel(), 256, [0,256], color='gray')
    else:  # 彩色图（仅显示Y通道）
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ax.hist(ycrcb[:,:,0].ravel(), 256, [0,256], color='gray')
    ax.set_title(title)
    ax.set_xlim([0,256])

# 读取图像
img = cv2.imread('lena.bmp', cv2.IMREAD_COLOR)
if img is None:
    # 如果找不到图片，从网络下载
    print("未找到本地lena.png，尝试从网络下载...")
    import urllib.request
    url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
    resp = urllib.request.urlopen(url)
    img = cv2.imdecode(np.frombuffer(resp.read(), np.uint8), cv2.IMREAD_COLOR)

# 转换为RGB格式用于显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 第一次直方图均衡化
equalized1 = equalize_image(img)
equalized1_rgb = cv2.cvtColor(equalized1, cv2.COLOR_BGR2RGB)

# 第二次直方图均衡化
equalized2 = equalize_image(equalized1)
equalized2_rgb = cv2.cvtColor(equalized2, cv2.COLOR_BGR2RGB)

# 创建画布
plt.figure(figsize=(15, 10))

# 显示原图
plt.subplot(3, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 2, 2)
plot_histogram(img, plt.gca(), 'Original Histogram')

# 显示第一次均衡化结果
plt.subplot(3, 2, 3)
plt.imshow(equalized1_rgb)
plt.title('After 1st Equalization')
plt.axis('off')

plt.subplot(3, 2, 4)
plot_histogram(equalized1, plt.gca(), '1st Equalized Histogram')

# 显示第二次均衡化结果
plt.subplot(3, 2, 5)
plt.imshow(equalized2_rgb)
plt.title('After 2nd Equalization')
plt.axis('off')

plt.subplot(3, 2, 6)
plot_histogram(equalized2, plt.gca(), '2nd Equalized Histogram')

plt.tight_layout()
plt.show()

# 保存结果
cv2.imwrite('original.png', img)
cv2.imwrite('equalized1.png', equalized1)
cv2.imwrite('equalized2.png', equalized2)