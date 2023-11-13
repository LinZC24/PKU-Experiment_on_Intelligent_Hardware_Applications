import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from picamera2 import Picamera2
import smbus
import time # 包含相关库文件
from time import sleep

address = 0x48
A0 = 0x40
bus = smbus.SMBus(1) # 初始化 i2c Bus
value = 0

Left = 1
Palm = 2
Five = 0
Right = 3

cam = Picamera2()
cam.still_configuration.main.size=(640,480)
cam.still_configuration.main.format='RGB888'
cam.configure("still")
#cam = cv2.VideoCapture(0)


# 神 经 网 络 同 训 练 时 的 网 络 结 构
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=9, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=5, stride=2)
        #self.fc1 = nn.Linear(in_features=12 * 3 * 2, out_features=60)
        #self.fc2 = nn.Linear(in_features=60, out_features=30)
        self.fc2 = nn.Linear(in_features=18 * 4 * 2, out_features=30)
        self.out = nn.Linear(in_features=30, out_features=4)
        #self.out = nn.Linear(in_features=60, out_features=4)

    def forward(self, t):
        t = F.relu(self.conv1(t))  # -> 38*28   76*56
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #t = F.max_pool2d(t, kernel_size=4, stride=4)  # ->19*14
        t = F.relu(self.conv2(t))  # -> 8*5
        t = F.max_pool2d(t, kernel_size=2, stride=2)  # ->4*2
        t = t.reshape(-1, 18 * 4 * 2)
        #t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        return self.out(t)


# def pic(img):
#     ret, frame = cv2.imread(img)
#     HSV = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV) # HSV颜 色 空 间 的 图 像
#     gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) # 灰 度 化 的 图 像
#     image_mask = cv2.inRange(HSV, np.array([0, 0, 0]), np.array([50, 200, 160])) # 实 验 室 相 机 的 肤 色
#     output = cv2.bitwise_and(gray, gray, mask=image_mask) # 按 颜 色 空 间 提 取 手
#     output = cv2.resize(output , (80, 60)) # 缩 小 图 片
#     output = cv2.blur(output , (2, 2)) # 模 糊 处 理
#     return output

cam.start()
time.sleep(1)

network = torch.load('network.pkl')

while True :
    frame = cam.capture_array('main')
    #output = pic(img)
    #ret, frame = cam.read()
    HSV = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV) # HSV颜 色 空 间 的 图 像
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) # 灰 度 化 的 图 像
    image_mask = cv2.inRange(HSV, np.array([2, 20, 30]), np.array([40, 200, 200])) # 实 验 室 相 机 的 肤 色
    output = cv2.bitwise_and(gray, gray, mask=image_mask) # 按 颜 色 空 间 提 取 手
    output = cv2.resize(output , (80, 60)) # 缩 小 图 片
    output = cv2.blur(output , (2, 2)) # 模 糊 处 理
    cv2.imshow('orig', frame) # 显 示 预 览
    cv2.imshow('gray', output)
    
    data = torch.tensor([[output]], dtype=torch.float) # 将 获 取 的 实 时 样 本 转 为 可 传 入 网络 的tensor
    pred_scores = network(data) # 获 取 各 类 型 的 分 数
    print(pred_scores)
    prediction = pred_scores.argmax(dim=1).item() # 取 最 大 者 为 结 果
    print(prediction)
#     cv2.imshow('img', img)
#     cv2.imshow('output', output)
    if prediction == Five and value == 0:
        value = 200
    elif prediction == Palm:
        value = 0
    elif prediction == Right and value > 0 and value < 250:
        value = value + 10
    elif prediction == Left and value > 10:
        value = value - 10
    print(value)
    bus.write_byte_data(address, A0, value)
    if cv2.waitKey(1) == ord("q"):
        
        break
    #sleep(0.2)   
cv2.destroyAllWindows()
#cam.release()
cam.stop()
# data = torch.tensor([[output]], dtype=torch.float) # 将 获 取 的 实 时 样 本 转 为 可 传 入 网络 的tensor
# pred_scores = network(data) # 获 取 各 类 型 的 分 数
# prediction = pred_scores.argmax(dim=1).item() # 取 最 大 者 为 结 果