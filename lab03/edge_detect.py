import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time
cam = Picamera2()
cam.still_configuration.main.size = (640, 480)
cam.still_configuration.main.format = 'RGB888'
cam.configure("still")
cam.start()
time.sleep(1)

while True:
    img = cam.capture_array('main')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    #cnt = contours[4]
    cv.drawContours(img, contours, -1, (0, 255, 0), 1)
    cv.imshow('thresh', thresh)
    cv.imshow('result' ,img)
    #cv.waitKey(0)
    #break
    if cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()
cam.release()