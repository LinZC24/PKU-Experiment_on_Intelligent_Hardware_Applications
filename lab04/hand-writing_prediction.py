import matplotlib.pyplot as plt
from sklearn import datasets , svm, metrics
import cv2 as cv
import numpy as np
from picamera2 import Picamera2, Preview
import time
from PIL import Image, ImageFont, ImageDraw

# 初始化摄像头
cam = Picamera2()
cam_config = cam.create_preview_configuration()
cam.configure(cam_config)
cam.start_preview(Preview.QTGL)
cam.start()
time.sleep(4)
cam.capture_file('original.jpg')
cam.stop_preview()

print('start training')

digits = datasets.load_digits() # 手 写 数 字 数 据 集 为 8x8 的 图 片， 16 级 灰 度

n_samples = len(digits.images)

# 将 二 维 数 据 变 成 一 维
data = digits.images.reshape((n_samples , -1))
#img_data = img.reshape((-1))
# 建 立 分 类 器
classifier = svm.SVC(kernel = 'linear', gamma='auto')

classifier.fit(data[:n_samples], digits.target[:n_samples])

expected = digits.target[:n_samples]

original = Image.open('original.jpg')

img = original.convert('L').resize((8, 8))
image_array = 16 - np.array(img) // 16 # 转化为16度灰度
imgToClassify = image_array.reshape(1, -1)# 转化为一维数据
print(imgToClassify)

# 预 测
predicted = classifier.predict(imgToClassify)

# 输出预测结果
plt.axis('off')
plt.imshow(img, cmap = plt.cm.gray_r, interpolation = 'nearest')
plt.title('prediction: %i' % predicted)
plt.show()