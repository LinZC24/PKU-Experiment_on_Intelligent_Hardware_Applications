import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time
cam = Picamera2() # 初始化摄像头
cam.still_configuration.main.size = (640, 480)
cam.still_configuration.main.format = 'RGB888'
cam.configure("still")
cam.start()
time.sleep(1)

while True:
    img = cam.capture_array('main') # 读取摄像头数据
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 转换颜色空间
    ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 寻找轮廓

    cv.drawContours(img, contours, -1, (0, 255, 0), 1) # 画出轮廓
    cv.imshow('thresh', thresh)
    cv.imshow('result' ,img)
    
    if cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()
cam.release()