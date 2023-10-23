import time
import spidev as SPI
import SSD1306
from PIL import Image, ImageFont, ImageDraw

RST = 19
DC = 16
bus = 0
device = 0

img = Image.new('RGB', (78, 39), 'black')
font = ImageFont.load_default()
draw = ImageDraw.Draw(img)

draw.text((0, 10), 'hello world!', font = font, fill = 255, align = 'center')
img = img.resize((128, 64))
img1 = img.convert('1')

disp = SSD1306.SSD1306(rst=RST,dc=DC,spi=SPI.SpiDev(bus,device))

disp.begin()
disp.clear()
disp.display()

disp.image(img1)
disp.display()