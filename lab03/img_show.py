import cv2 as cv
img = cv.imread('code\lab03\watch.jpg', 1)
cv.imshow('card', img)
cv.waitKey(0)
cv.destroyWindow('card')