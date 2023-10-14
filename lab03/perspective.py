import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import time
original_points = np.float32([[100, 50], [540, 0], [0, 380], [440, 380]])
target_points = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
perspective_matrix = cv.getPerspectiveTransform(original_points, target_points)
cam = Picamera2()
cam.still_configuration.main.size = (640, 480)
cam.still_configuration.main.format = 'RGB888'
cam.configure("still")
cam.start()
time.sleep(1)

while True:
    img = cam.capture_array('main')
    output = cv.warpPerspective(img, perspective_matrix, (640, 480))
    cv.imshow('Original', img)
    cv.imshow('Output', output)
    cv.waitKey(0)
    break
cv.destroyAllWindows()
cam.release()