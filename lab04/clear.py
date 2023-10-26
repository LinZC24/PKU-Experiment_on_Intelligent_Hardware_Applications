import time
import spidev as SPI
import SSD1306
from PIL import Image

RST = 19
DC = 16
bus = 0
device = 0 
disp = SSD1306.SSD1306(rst=RST,dc=DC,spi=SPI.SpiDev(bus,device))

disp.begin()
disp.clear()
disp.display() # 初 始 化 屏 幕 相 关 参 数 及 清 屏