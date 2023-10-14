import cv2 as cv
img = cv.imread('watch.jpg', 1)
cv.imshow('watch', img)
cv.waitKey(0)
cv.destroyWindow('watch')