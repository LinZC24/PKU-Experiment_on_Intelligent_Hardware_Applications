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
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img, 5)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius= 30, maxRadius=50)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
      cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
      cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3) 

    cv.imshow('original', img)
    cv.imshow('result', cimg)
    if cv.waitKey(1) == ord('q'):
        break
cv.destroyAllWindows()
cam.release()