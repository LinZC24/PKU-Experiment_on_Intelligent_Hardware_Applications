import time
import spidev as SPI
import SSD1306
from PIL import Image

# 设置树莓派管脚信息
RST = 19
DC = 16
bus = 0
device = 0 
disp = SSD1306.SSD1306(rst=RST,dc=DC,spi=SPI.SpiDev(bus,device))

disp.begin()
disp.clear()
disp.display() # 初始化屏幕相关参数及清屏

image = Image.open('pku.png').resize((disp.width , disp.height),Image.ANTIALIAS).convert('1')
disp.image(image)
disp.display()