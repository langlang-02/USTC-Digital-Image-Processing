import numpy as np
import cv2
from scipy.fftpack import dct, idct
from scipy.linalg import hadamard

# 定义矩阵
matrix = np.array([
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0]
], dtype=np.float32)

# 计算DFT
dft = cv2.dft(matrix, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_magnitude = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])

# 计算DCT
dct_matrix = dct(dct(matrix.T, norm='ortho').T, norm='ortho')

# 计算Hadamard变换
# 确保矩阵大小为2的幂次
hadamard_matrix = hadamard(4)
hadamard_transform = np.dot(np.dot(hadamard_matrix, matrix), hadamard_matrix)

# 计算Haar变换
def haar_transform(matrix):
    n = matrix.shape[0]
    output = matrix.copy()
    h = 1
    while h < n:
        for i in range(0, n, h*2):
            for j in range(0, h):
                a = output[i+j]
                b = output[i+j+h]
                output[i+j] = (a + b) / np.sqrt(2)
                output[i+j+h] = (a - b) / np.sqrt(2)
        h *= 2
    return output

haar_matrix = haar_transform(matrix)

# 打印结果
print("DFT Magnitude:\n", dft_magnitude)
print("\nDCT:\n", dct_matrix)
print("\nHadamard Transform:\n", hadamard_transform)
print("\nHaar Transform:\n", haar_matrix)
