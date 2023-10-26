import numpy as np
import time
import spidev as SPI
import SSD1306
from PIL import Image, ImageFont, ImageDraw
from sklearn import datasets , svm, metrics

RST = 19
DC = 16
bus = 0
device = 0 
disp = SSD1306.SSD1306(rst=RST,dc=DC,spi=SPI.SpiDev(bus,device))

font = ImageFont.load_default()

digits = datasets.load_digits() # 手 写 数 字 数 据 集 为 8x8 的 图 片， 16 级 灰 度
# digits.images 是 图 片 数 据， digits.target 是 标 签
images_and_labels = list(zip(digits.images , digits.target))

n_samples = len(digits.images)
# 将 二 维 数 据 变 成 一 维
data = digits.images.reshape((n_samples , -1))

# 建 立 分 类 器
classifier = svm.SVC(kernel = 'rbf', gamma=0.001)

# 用 前 一 半 数 据 进 行 训 练
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

expected = digits.target[n_samples // 2:]
# 预 测
predicted = classifier.predict(data[n_samples // 2:])

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    #digit = Image.fromarray((kk*8).astype(np.uint8), mode='L').resize((48,48)).convert('1')
    digit = Image.fromarray((image*8).astype(np.uint8), mode='L').resize((48,48)).convert('1')
    img = Image.new('1',(disp.width ,disp.height),'black')
    img.paste(digit , (0, 8))
    pre = Image.new('1', (24, 24), 'black')
    draw = ImageDraw.Draw(pre)
    #print(type(prediction))
    #t = prediction
    draw.text((0, 0), str(prediction), font = font, fill = 255)
    pre = pre.resize((48, 48))
    img.paste(pre, (64, 8))
    disp.clear()
    disp.display()
    disp.image(img)
    disp.display()
    input('press a key to continue')
disp.clear()