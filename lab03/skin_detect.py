import numpy as np
import cv2 as cv
from picamera2 import Picamera2
import time
cam = Picamera2()
cam.still_configuration.main.size = (640, 480)
cam.still_configuration.main.format = 'RGB888'
cam.configure("still")
cam.start()
time.sleep(1)

while True:
    frame = cam.capture_array('main')
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    image_mask = cv.inRange(hsv, np.array([0, 50, 0]), np.array([255, 200, 255]))# get skin
    output = cv.bitwise_and(frame, frame, mask = image_mask)
    cv.imshow('Original', frame)
    cv.imshow('Output', output)
    if cv.waitKey(1) == ord("q"):
        break
cv.destroyAllWindows()
cam.release()