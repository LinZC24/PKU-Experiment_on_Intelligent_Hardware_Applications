import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture('pingpong.mp4')
ret, frame = cap.read()

r, h, c, w = 288, 140, 100, 140
track_window = (c, r, w, h)
roi = frame[r:r+h, c:c+w]
hsv_roi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0, 0, 180)), np.array((180, 255, 255)))
# calculating object histogram
roi_hist = cv.calcHist([hsv_roi],[0], None, [180], [0, 180] )
# normalize histogram and apply backprojection
cv.normalize(roi_hist ,roi_hist ,0,255,cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
#dst = cv.calcBackProject([hsvt],[0,1],roihist ,[0,180,0,256],1)
while True:
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv.CamShift(dst, track_window , term_crit)

        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img = cv.polylines(frame ,[pts],True, 255,2)
        cv.imshow('img',img)
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img)
    else:
        break

cv.destroyAllWindows()
cap.release()