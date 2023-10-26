import cv2 as cv
import numpy as np
img = cv.imread("lab03/watch.jpg", 1)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img_mask = cv.inRange(hsv, np.array([0, 43, 46]), np.array([200, 200, 200]))

output = cv.bitwise_and(img, img, mask = img_mask)
cv.imshow("mask", img_mask)
cv.imshow("original", img)
cv.imshow("output", output)
cv.waitKey(0)