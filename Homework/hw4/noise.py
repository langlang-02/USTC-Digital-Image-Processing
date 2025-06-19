import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取 Lena 图像
image = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

# 高斯噪声的添加
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = np.clip(image + gauss, 0, 255)  # 添加噪声并确保像素值在0到255之间
    #clip钳位操作，确保像素值在0到255之间
    return noisy.astype(np.uint8)

# 椒盐噪声的添加
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = image.copy()
    row, col = image.shape
    # 添加盐噪声
    salt = np.random.rand(row, col) < salt_prob
    noisy[salt] = 255  # 盐噪声将像素设为255
    # 添加胡椒噪声
    pepper = np.random.rand(row, col) < pepper_prob
    noisy[pepper] = 0  # 胡椒噪声将像素设为0
    return noisy

# 局域平均滤波
def local_average_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

# 中值滤波
def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# 处理图像
# 1. 高斯噪声处理
gaussian_noisy = add_gaussian_noise(image)
gaussian_average_filtered = local_average_filter(gaussian_noisy)
gaussian_median_filtered = median_filter(gaussian_noisy)

# 2. 椒盐噪声处理
salt_and_pepper_noisy = add_salt_and_pepper_noise(image)
salt_and_pepper_average_filtered = local_average_filter(salt_and_pepper_noisy)
salt_and_pepper_median_filtered = median_filter(salt_and_pepper_noisy)

# 显示结果
fig, axs = plt.subplots(2, 4, figsize=(12, 9))

# 原图
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# 高斯噪声
axs[0, 1].imshow(gaussian_noisy, cmap='gray')
axs[0, 1].set_title('Gaussian Noisy Image')
axs[0, 1].axis('off')

# 高斯噪声平均滤波
axs[0, 2].imshow(gaussian_average_filtered, cmap='gray')
axs[0, 2].set_title('Gaussian Average Filtered')
axs[0, 2].axis('off')

# 高斯噪声中值滤波
axs[0, 3].imshow(gaussian_median_filtered, cmap='gray')
axs[0, 3].set_title('Gaussian Median Filtered')
axs[0, 3].axis('off')


axs[1, 0].axis('off')

# 椒盐噪声
axs[1, 1].imshow(salt_and_pepper_noisy, cmap='gray')
axs[1, 1].set_title('Salt and Pepper Noisy Image')
axs[1, 1].axis('off')

# 椒盐噪声平均滤波
axs[1, 2].imshow(salt_and_pepper_average_filtered, cmap='gray')
axs[1, 2].set_title('Salt and Pepper Average Filtered')
axs[1, 2].axis('off')

# 椒盐噪声中值滤波
axs[1, 3].imshow(salt_and_pepper_median_filtered, cmap='gray')
axs[1, 3].set_title('Salt and Pepper Median Filtered')
axs[1, 3].axis('off')

plt.tight_layout()


# 保存图像为文件
plt.savefig('denoised_image_results.png')  # 保存为PNG文件，可以选择其他格式，如 .jpg, .pdf 等

plt.show()