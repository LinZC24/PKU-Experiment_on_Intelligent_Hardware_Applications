import cv2
import numpy as np
import time
from picamera2 import Picamera2

cam = Picamera2()
cam.still_configuration.main.size = (640,480)
cam.still_configuration.main.format = 'RGB888'
cam.configure("still")
cam.start()
time.sleep(1) 

def mask_detect(img):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(eyes) == 0:
        return
    for (x, y, w, h) in eyes:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    mask_x_begin = x
    mask_y_begin = y
    mask_x_end = x + w
    mask_y_end = y + h
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, np.array([100, 100, 50]),np.array([140,255,255]))
    
    skin_mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
    skin_mask[mask_y_begin: mask_y_end, mask_x_begin: mask_x_end] = skin[mask_y_begin: mask_y_end, mask_x_begin: mask_x_end]
    out = cv2.bitwise_and(img, img, mask = skin_mask)
    mask = np.where((out == 0), 0, 1).astype('uint8')
    mask_area = mask.sum()
    
    if mask_area /  ((mask_x_end - mask_x_begin) * (mask_y_end - mask_y_begin)) > 0.8:
        print(mask_area /  ((mask_x_end - mask_x_begin) * (mask_y_end - mask_y_begin)))
        cv2.putText(img, "Mask Detected", (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    else:
        print(mask_area /  ((mask_x_end - mask_x_begin) * (mask_y_end - mask_y_begin)))
        cv2.putText(img, "No Mask Detected", (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
while True:
    img = cam.capture_array('main')
    mask_detect(img)
    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break
    
     
cv2.destroyAllWindows()
cam.stop()
