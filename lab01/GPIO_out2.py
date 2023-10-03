import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM) # 设置采用BCM编号

light = 26
GPIO.setup(light, GPIO.OUT) # 初始化引脚
p = GPIO.PWM(light, 50) # 创建一个PWM实例
p.start(0) # 指定初始占空比为0

try:
    while True:
        for dc in range(0, 101, 2): # 由暗变亮，通过增加占空比实现
            p.ChangeDutyCycle(dc)
            time.sleep(0.02)
        for dc in range(100, -1, -2): # 由亮变暗，通过降低占空比实现
            p.ChangeDutyCycle(dc)
            time.sleep(0.02)
except KeyboardInterrupt:
    pass
    p.stop()
    GPIO.cleanup() # 按下ctrl+C时释放引脚
finally:
    pass
    p.stop()
    GPIO.cleanup() # 确保释放引脚