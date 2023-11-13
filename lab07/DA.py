import smbus
import time # 包含相关库文件
address = 0x48
A0 = 0x40
bus = smbus.SMBus(1) # 初始化 i2c Bus
value = 30
while True:
    bus.write_byte_data(address, A0, value) # 循环写入
    #print("writing")
    