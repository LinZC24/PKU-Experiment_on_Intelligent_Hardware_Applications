import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM) # 设置采用BCM编号

key = 20
light = 26
lighter = True # 借助布尔值lighter判断目前LED灯的明暗变化方向
GPIO.setup(light, GPIO.OUT)
GPIO.setup(key, GPIO.IN, GPIO.PUD_UP) # 初始化引脚
p = GPIO.PWM(light, 50) # 创建PWM实例
p.start(0)

def my_callback(ch): # 定义my_callback函数
    global lighter
    if lighter == True:
        lighter = False
    else:
        lighter = True
GPIO.add_event_detect(key, GPIO.RISING, callback = my_callback, bouncetime = 200) # 设置发生指定事件时刻的回调函数，参数中bouncetime用于消抖

try: # 控制lED灯明灭的代码逻辑与呼吸灯的实现相同
    while True:
        if lighter == True:
            for dc in range(0, 101, 2):
                p.ChangeDutyCycle(dc)
                time.sleep(0.05)
            p.ChangeDutyCycle(0)
        elif lighter == False:
            for dc in range(100, -1, -2):
                p.ChangeDutyCycle(dc)
                time.sleep(0.05)
            p.ChangeDutyCycle(100)
finally:
    p.stop()
    GPIO.cleanup() # 确保释放引脚