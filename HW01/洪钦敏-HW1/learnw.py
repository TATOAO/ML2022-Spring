import numpy as np

w = 1.2
b = 0.5

dataset = [
    [0.2, 1],
    [0.7, 0.25],
    [0.6, 1.1],
    [1.2, 5.2],
    [1.3, 3.1],
]

origin = np.array(dataset)
x = origin[:, 0]
y = origin[:, 1]
print(f'x:{x} y:{y}')
# 计算第一次的y
y1 = w * x + b
print('y1', y1)
size = y.size
mse = ((y1 - y) ** 2).sum() / size
print(mse)
# cost函数求导

dw = 2 * (w * (x ** 2).sum() - ((y - b) * x).sum())
db = 2 * (size * b - ((y - w * x).sum()))
print(f'dw:{dw} db:{db}')
num = 0
lr = 1e-3
while dw != 0 or db != 0:
    w = w - lr * dw
    b = b - lr * db
    dw = 2 * (w * (x ** 2).sum() - ((y - b) * x).sum())
    db = 2 * (size * b - ((y - w * x).sum()))
    num += 1
    if num > 10000:
        break

print(num)
print(w)
print(b)
mse = ((w * x + b - y) ** 2).sum() / size
print(mse)
