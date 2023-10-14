import cv2 as cv
import numpy as np
img = cv.imread('watch.jpg', 1)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
image_mask = cv.inRange(hsv, np.array([40,50,50]), np.array([50,200,200]))

output = cv.bitwise_and(img, img, mask = image_mask)
cv.imshow("Original", img)
cv.imshow("Output", output)
cv.waitKey(0)

cv.destroyAllWindows