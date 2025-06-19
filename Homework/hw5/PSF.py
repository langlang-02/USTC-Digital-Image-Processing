import numpy as np
import matplotlib.pyplot as plt

L = 10  # 模糊长度（像素）
x = np.linspace(-L, L, 1000)
h = np.where(np.abs(x) <= L/2, 1/L, 0)  # PSF

plt.figure(figsize=(8, 4))
plt.plot(x, h, 'b-', linewidth=2, label=f'PSF (L={L})')
plt.xlabel('x (pixels)')
plt.ylabel('h(x, 0)')
plt.title('Horizontal Motion Blur PSF')
plt.axvline(x=-L/2, color='r', linestyle='--')
plt.axvline(x=L/2, color='r', linestyle='--')
plt.legend()
plt.grid(True)

plt.savefig('movePSF.png',transparent=True)
plt.show()