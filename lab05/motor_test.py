import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
GPIO.setup(12, GPIO.OUT) # 使用 12 号管脚来控制舵机
pwm = GPIO.PWM(12,50)
pwm.start(0)

while True:
    a = input()
    print(a)
    a = int(a)
    pwm.ChangeDutyCycle(a) # 输出 90 度
    sleep(1)
pwm.stop()
GPIO.cleanup()