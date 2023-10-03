import cv2 as cv
import numpy as np
img = cv.imread('code\lab03\watch.jpg', 1)
size = img.shape
ogp = [[0, 0], [size[0] - 1, 0], [0, size[1] - 1], [size[0] - 1, size[1] - 1]]

original_points = np.float32([ogp[0], ogp[1], ogp[2], ogp[3]])
target_points = np.float32([ogp[0], ogp[1], ogp[2], ogp[3]])
perspective_matrix = cv.getPerspectiveTransform(original_points, target_points)
output_img = cv.warpPerspective(img, perspective_matrix, (size[1], size[0]))
cv.imshow('i', output_img)
cv.imshow('origin', img)
cv.waitKey(0)