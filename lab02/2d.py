import matplotlib.pyplot as plt
import numpy as np

# 进行生成数据，拼接初始数据等处理
r1 = 2
r2 = 4
p1 = []
p2 = []
p = np.array([[np.random.rand() * r2 * 2 - r2, np.random.rand() * r2 * 2 - r2]])
for i in range(10000):
    x = np.random.rand() * r2 * 2 - r2;
    y = np.random.rand() * r2 * 2 - r2;
    if abs(x * x + y * y - r1) < 0.5:
        p1.append([x, y])
    if abs(x * x + y * y - r2) < 0.5:
        p2.append([x, y])
    p = np.append(p, np.array([[x, y]]), axis = 0)
t1 = np.mean(p1, axis = 0)
t2 = np.mean(p2, axis = 0)

# 维护新聚点的函数
def change(tp):
    r = np.mean(tp, axis = 0)
    return r

m , n = np.array([p[0]]), np.array([p[1]])
result1, result2 = m[0], n[0]

# 二维数据聚类算法
for i in range(2, int(p.size / 2)):
    if abs(np.sqrt(np.sum(np.square(p[i] - t1))) - r1) < abs(np.sqrt(np.sum(np.square(p[i] - t2))) - r2):
        if abs(np.sqrt(np.sum(np.square(p[i] - t1))) - r1) < 0.5:
          m = np.append(m, np.array([p[i]]), axis = 0)
          result1 = change(m)
    else:
        if abs(np.sqrt(np.sum(np.square(p[i] - t2))) - r2) < 0.5:
          n = np.append(n, np.array([p[i]]), axis = 0)
          result2 = change(n)

# 数据可视化及输出
plt.scatter(m[:,0], m[:,1], c='r', s=3)
plt.scatter(n[:,0], n[:,1], c='b', s=3)
plt.show()
print(t1, t2, result1, result2)