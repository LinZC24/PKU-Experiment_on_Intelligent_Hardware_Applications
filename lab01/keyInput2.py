import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM) # 设置采用BCM编号

key = 20
led = 26
freq = 1
t = 0
stop = False
light = False

GPIO.setup(led, GPIO.OUT, initial = GPIO.LOW)
GPIO.setup(key, GPIO.IN, GPIO.PUD_UP)

def my_callback(ch): # 定义my_callback函数,在按键时调用
    global t
    global stop
    global freq
    global light
    if stop == False and GPIO.input(led) == GPIO.LOW: # 判断初始状态，开始闪烁
        light = True
    else:
        freq = freq / 2 # 每次单击闪烁频率加倍
    if (time.time() - t) < 0.5: # 若两次点击时间小于0.5s，则判断为双击
        stop = True
        light = False
    else:
        t = time.time()
GPIO.add_event_detect(key, GPIO.RISING, callback = my_callback, bouncetime = 50) # 回调函数，将bouncetime设置得较短防止因消抖时间过长影响双击的判断

try:
    while True:
        if light:
            GPIO.output(led, GPIO.HIGH)
            time.sleep(freq)
            GPIO.output(led, GPIO.LOW)
            time.sleep(freq)
        elif stop:
            GPIO.output(led, GPIO.LOW)
            freq = 2
except KeyboardInterrupt:
    GPIO.cleanup() # 按下ctrl+C时释放引脚
finally:
    GPIO.cleanup() # 确保释放引脚