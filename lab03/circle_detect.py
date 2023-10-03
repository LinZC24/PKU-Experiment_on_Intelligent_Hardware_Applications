import cv2 as cv
import numpy as np

img = cv.imread('code\lab03\watch.jpg', 0)
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius= 30, maxRadius=50)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
  cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
  cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3) 

cv.imshow('original', img)
cv.imshow('result', cimg)
cv.waitKey(0)