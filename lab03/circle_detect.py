import cv2 as cv
import numpy as np

img = cv.imread('watch.jpg', 0)
img = cv.medianBlur(img, 5) # 图像滤波
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) # 转化为灰度图片

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=40, minRadius= 30, maxRadius=50) 
#　使用HoughCircles（）寻找圆形

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
  cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
  cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3) 
#　在图片中将圆形标记出来

cv.imshow('original', img)
cv.imshow('result', cimg)
cv.waitKey(0)