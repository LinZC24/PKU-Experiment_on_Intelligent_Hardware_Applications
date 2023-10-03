import cv2 as cv
img = cv.imread('lab03\wx20231002002220.jpg', 1)
cv.imshow('card', img)
cv.waitKey(0)
cv.destroyWindow('card')