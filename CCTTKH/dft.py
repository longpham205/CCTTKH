import numpy as np

def dft2d(image):
    F = np.fft.fft2(image)
    return F

def idft2d(F):
    f = np.fft.ifft2(F)
    return np.round(f.real).astype(int)

block = np.array([
    [98, 92, 95, 80, 75, 82, 68, 50],
    [97, 91, 94, 79, 74, 81, 67, 49],
    [95, 89, 92, 77, 72, 79, 65, 47],
    [93, 87, 90, 75, 70, 77, 63, 45],
    [91, 85, 88, 73, 68, 75, 61, 43],
    [89, 83, 86, 71, 66, 73, 59, 41],
    [87, 81, 84, 69, 64, 71, 57, 39],
    [85, 79, 82, 67, 62, 69, 55, 37]
])

F = dft2d(block)
print("Biến đổi Fourier 2D (phần thực):")
print(np.round(F.real, 2))

restored = idft2d(F)
print("\nKhối ảnh khôi phục sau IDFT:")
print(restored)