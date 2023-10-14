import cv2 as cv
import numpy as np
i = 0
original_points = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
target_points = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
img = cv.imread('card.png')
cv.imshow('img', img)
def mouse(event, x, y, flags, param):
    global i
    if event == cv.EVENT_LBUTTONDOWN:
        if i == 4:
            perspective_matrix = cv.getPerspectiveTransform(original_points, target_points)
            output = cv.warpPerspective(img, perspective_matrix, (640, 480))        
            cv.imshow('Output', output)
        print(x, y)
        original_points[i] = [x, y]
        i = i + 1
cv.setMouseCallback('img', mouse)
cv.waitKey(0)
cv.destroyAllWindows()

