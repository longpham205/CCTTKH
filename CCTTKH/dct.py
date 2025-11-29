import numpy as np
from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

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

print("Mảng ban đầu:")
print(block)

F = dct2(block)
print("\nKết quả DCT:")
print(np.round(F, 2))

restored = idct2(F)
print("\nKhối ảnh khôi phục sau IDCT:")
print(np.round(restored).astype(int))