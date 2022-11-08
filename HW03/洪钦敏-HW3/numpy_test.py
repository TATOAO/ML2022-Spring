import numpy as np
from numpy import random

# 测试 最简单的Ensemble  权重 这个dot有点难以理解
# batch_size 为2    一共3个结果 每个结果5个分类
x = random.randint(1, 5, size=(2, 3, 5))
print(x.shape)
print(x)
# weight = np.array([[0.3, 0.3, 0.4],[0.3, 0.3, 0.4]])
# 最后一个结果0.4权重 其他0.3
weight = np.array([0.3, 0.3, 0.4])
# 权重在前
y = np.dot(weight, x)
print(y.shape)
print(y)
z = x.sum(axis=1)
print(z.shape)
print(z)
