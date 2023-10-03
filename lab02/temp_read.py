import glob
import os

for name in glob.glob('/sys/bus/w1/devices/2*'): # 利用通配符查找满足要求的文件夹
    folder = name + '/w1_slave' # 拼接字符串获得目标文件路径
    with open(folder, 'r') as f: #利用open()函数打开对应文件并输出
        for line in f:
            print(line)