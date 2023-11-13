import smbus
import time # 包含相关库文件
address = 0x48
A0 = 0x40
bus = smbus.SMBus(1) # 初始化 i2c Bus
value = bus.read_byte(address) # 读出
while True:
    value = value - 5 # darking
    print(value)
    bus.write_byte_data(address, A0, value) # 循环写入
    time.sleep(1)
    
