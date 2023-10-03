import RPi.GPIO as GPIO
import time
import random #导入random库，方便之后使用random.choices函数以根据历史数据指定出圈策略

GPIO.setmode(GPIO.BCM) # 设置采用BCM编号

stone = 5
scissors = 6
cloth = 13
led = 26
select_stone = 0
select_scissors = 0
select_cloth = 0
select = -1

GPIO.setup(stone, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(scissors, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(cloth, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(led, GPIO.OUT, initial = GPIO.LOW)

def my_callback(ch):
    global select_stone
    global select_scissors
    global select_cloth
    global select
    # 根据玩家选项，记录玩家出拳的历史数据，为指定策略提供依据
    if ch == stone:
        select_stone = select_stone + 1
        select = 0
    elif ch == scissors:
        select_scissors = select_scissors + 1
        select = 1
    elif ch == cloth:
        select_cloth = select_cloth + 1
        select = 2

# 按下不同按键的回调函数
GPIO.add_event_detect(stone, GPIO.RISING, callback = my_callback, bouncetime = 200)
GPIO.add_event_detect(scissors, GPIO.RISING, callback = my_callback, bouncetime = 200)
GPIO.add_event_detect(cloth, GPIO.RISING, callback = my_callback, bouncetime = 200)

try:
    print('push a button')
    while True:     
        if select != -1:
            ai_select = random.choices([0, 1, 2], [select_scissors, select_cloth, select_stone], k = 1) # 根据历史数据，增加相应选项的权重，判断玩家出拳习惯，增加电脑获胜几率
            if ai_select[0] == 0:
                if select == 0:
                    print('your select is stone, ai\'s select is stone, the result is draw!')
                elif select == 1:
                    print('your select is scissors, ai\'s select is stone, you lose!')
                elif select == 2:
                    print('your select is cloth, ai\'s select is stone, you win!')
            elif ai_select[0] == 1:
                if select == 0:
                    print('your select is stone, ai\'s select is scissors, you win!')
                elif select == 1:
                    print('your select is scissors, ai\'s select is scissors, the result is draw!')
                elif select == 2:
                    print('your select is cloth, ai\'s select is scissors, you lose!')
            elif ai_select[0] == 2:
                if select == 0:
                    print('your select is stone, ai\'s select is cloth, you lose!')
                elif select == 1:
                    print('your select is scissors, ai\'s select is cloth, you win!')
                elif select == 2:
                    print('your select is cloth, ai\'s select is cloth, the result is draw!')
            select = -1
except KeyboardInterrupt:
    GPIO.cleanup() # 按下ctrl+C时释放引脚
finally:
    GPIO.cleanup() # 确保释放引脚