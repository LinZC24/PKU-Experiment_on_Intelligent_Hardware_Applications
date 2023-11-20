import cv2 as cv
import numpy as np
i = 0
original_points = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
target_points = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
img = cv.imread('card.png')
cv.imshow('img', img)
def mouse(event, x, y, flags, param): # 定义处理鼠标动作的函数，在回调函数中调用，选取图片上的点作为透视变换的初始区域
    global i
    if event == cv.EVENT_LBUTTONDOWN:
        if i == 4:
            perspective_matrix = cv.getPerspectiveTransform(original_points, target_points) # 计算透视变换矩阵
            output = cv.warpPerspective(img, perspective_matrix, (640, 480))  # 进行透视变换      
            cv.imshow('Output', output)
        print(x, y)
        original_points[i] = [x, y]
        i = i + 1
cv.setMouseCallback('img', mouse) #回调函数
cv.waitKey(0)
cv.destroyAllWindows()

