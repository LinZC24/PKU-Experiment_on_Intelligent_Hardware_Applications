import cv2 as cv
img = cv.imread('lab03/watch.jpg', 1)
cv.imshow('watch', img)
cv.waitKey(0)
cv.destroyWindow('watch')