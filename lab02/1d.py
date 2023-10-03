import glob
import numpy as np
import matplotlib.pyplot as plt

# 随机生成两组不同的满足正态分布的数据并将它们拼接起来
t1 = np.random.randn(360)
t2 = np.random.randn(360)
x1 = np.random.randint(20, 28)
x2 = np.random.randint(20, 28)
temp = np.append(t1, t2)

# 每次添加新元素后维护聚点，返回值为新的聚点
def change(s):
    m = np.mean(s)
    return m

# 初始聚点随机选取
k1, k2 = [temp[0] + x1], [temp[361] + x2]
m1, m2 = k1[0], k2[0]

# 遍历及分类
for i in range(2, temp.size):
    if abs(temp[i] - m1 + x1) < abs(temp[i] - m2 + x2):
        k1.append(temp[i] + x1)
        m1 = change(k1)
    elif abs(temp[i] - m1 + x1) > abs(temp[i] - m2 + x2):
        k2.append(temp[i] + x2)
        m2 = change(k2)

# 数据以直方图的形式进行可视化输出，同时输出分类结果，与已知均值比较
plt.hist(k1, bins = 360, facecolor = 'r', alpha = 0.5)
plt.hist(k2, bins = 360, facecolor = 'b', alpha = 0.5)
plt.show()
print('x1:', x1,'x2:', x2)
print('m1:', m1,'m2:',m2)
