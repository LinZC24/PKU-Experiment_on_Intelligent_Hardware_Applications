import cv2
import time
from picamera2 import Picamera2

cam = Picamera2()
cam.still_configuration.main.size = (640,480)
cam.still_configuration.main.format = 'RGB888'
cam.configure("still")
cam.start()
time.sleep(1) 

while True:
    img = cam.capture_array('main')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.namedWindow('face detected')
    cv2.imshow('Face Detected!', img)
    
    if cv2.waitKey(1) == ord('q'):
        break
     
cv2.destroyAllWindows()
cam.release()