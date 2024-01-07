import numpy as np
import cv2 as cv
from picamera2 import Picamera2
import time

from serial.tools import list_ports
from pydobot import Dobot
from arm import Arm
from detect import vision

cam = Picamera2()
cam.still_configuration.main.size = (640, 480)
cam.still_configuration.main.format = 'RGB888'
cam.configure("still")
cam.start()
time.sleep(1)

v = vision()

while True:
    img = cam.capture_array("main")
    v.detect_edge(img)
    #print(f'x:{x0}, y:{y0}')
    v.detect_color(img)
    cv.imshow("result", img)
    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
cam.release()