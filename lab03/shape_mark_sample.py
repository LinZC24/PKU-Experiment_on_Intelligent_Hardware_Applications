import numpy as np
import cv2 as cv

im = cv.imread('lab03/pods.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

#cnt = contours[4]
cv.drawContours(im, contours, -1, (0, 255, 0), 1)
cv.imshow('thresh', thresh)
cv.imshow('result' ,im)
cv.waitKey(0)