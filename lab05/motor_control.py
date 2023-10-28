from pyswip import Prolog, registerForeign
import RPi.GPIO as GPIO
from time import sleep
import smbus
import time

key = 20
address = 0x20
bus = smbus.SMBus(1)
GPIO.setmode(GPIO.BCM)
GPIO.setup(key, GPIO.IN, GPIO.PUD_UP)
GPIO.setup(12, GPIO.OUT) # 使用 12 号管脚来控制舵机
pwm = GPIO.PWM(12,50) # 创建pwm对象
animal_dic = {}
prolog =Prolog()
init = 0
def my_callback(ch):
    global init
    init = 1
    print('back')
    
GPIO.add_event_detect(key, GPIO.RISING, callback = my_callback, bouncetime = 200) # 回调函数，每次按下五向摇杆的按键后将舵机的位置复位
# 利用Python重新定义verify函数
def verify(s):
    if s in animal_dic:
        return animal_dic[s]
    else:
        sleep(1)
        print("Does the animal have the following attribute: ", s, "?")
        while True:
            a = bus.read_byte(address)
            
            if a == 254 or a == 247:
                r = 'yes'
                print('yes')
                break
            elif a == 253 or a == 251:
                r = 'no'
                print('no')
                break

        if r.lower() in ['yes', 'y']:
            animal_dic[s] = True
            return True
        elif r.lower() in ['no', 'n']:
            animal_dic[s] = False
            return False
        else:
            print('please enter the answer \'yes\' or \'no\'')
            return verify(s)
verify.arity = 1
registerForeign(verify)
prolog.consult('animal.pl')
pwm.start(0)
sleep(1)
# 创建循环，使询问的过程可以重复
while True:
    animal_dic.clear() # 清空字典，开始新的一次查询
    
    for result in prolog.query('hypothesize(X)'):
        if init == 1:
            pwm.ChangeDutyCycle(5.5)
            sleep(2)
            init = 0
            print('init the motor')
        print("I guess the animal is:", result["X"])
        
        if result["X"] == 'ostrich':
            pwm.ChangeDutyCycle(9)
            sleep(1)            
        elif result["X"] == 'giraffe':
            pwm.ChangeDutyCycle(8)
            sleep(1)            
        elif result["X"] == 'tiger':
            pwm.ChangeDutyCycle(7)
            sleep(1)            
        elif result["X"] == 'zebra':
            pwm.ChangeDutyCycle(6)
            sleep(1)            
        elif result["X"] == 'albatross':
            pwm.ChangeDutyCycle(5)
            sleep(1)            
        elif result["X"] == 'penguin':
            pwm.ChangeDutyCycle(4)
            sleep(1)            
        elif result["X"] == 'cheetah':
            pwm.ChangeDutyCycle(3)
            sleep(1)            
        else:
            pwm.ChangeDutyCycle(2)
            sleep(1)
pwm.stop()
GPIO.cleanup()