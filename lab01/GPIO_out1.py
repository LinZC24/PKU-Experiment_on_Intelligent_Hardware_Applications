import RPi.GPIO as GPIO 
import time
GPIO.setmode(GPIO.BCM) # 设置采用BCM编号

light = 26
GPIO.setup(light, GPIO.OUT, initial = GPIO.LOW) # 初始化引脚

try:
    while True:
        GPIO.output(light, GPIO.HIGH) # 引脚设置为高电位，LED点亮
        time.sleep(0.2) 
        GPIO.output(light, GPIO.LOW)
        time.sleep(0.2) # 利用time.sleep()达到闪烁的效果
except KeyboardInterrupt:
    GPIO.cleanup() # 按下ctrl+C时释放引脚
finally:
    GPIO.cleanup() # 确保释放引脚