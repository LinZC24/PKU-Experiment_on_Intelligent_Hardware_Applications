import numpy as np
import cv2 as cv
import time

class vision:
    red_low = np.array([0, 50,100])
    red_high = np.array([15, 255, 255])
    blue_low = np.array([75, 50, 100])
    blue_high = np.array([124, 255, 255])
    yellow_low = np.array([25, 45, 45])
    yellow_high = np.array([35, 255, 255])
    green_low = np.array([35, 45, 45])
    green_high = np.array([75, 255, 255])
        
    def detect_edge(self, img):
        image = cv.medianBlur(img, 5)
        imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 2)
        cv.imshow('t', thresh)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        for i in contours:
            area = cv.contourArea(i)
            cv.drawContours(img, i, -1, (255, 0, 0), 2)
            if area > 1000 and area < 100000:
                cv.drawContours(img, i, -1, (0, 255, 0), 2)
                p = cv.arcLength(i, True)
                approx = cv.approxPolyDP(i, 0.1 * p, True)
                n = len(approx)
                x, y, w, h = cv.boundingRect(approx)
                #print(x, y)
                if n == 3:
                    contour_type = "triangle"
                elif n == 4:
                    contour_type = "rectangle"
                elif n >= 5:
                    contour_type = "circle"
                else:
                    contour_type = "none"        
                cv.putText(img, contour_type, (x + (w // 2), y + 2 * (h // 3)), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                #return x + w//2 , y + h//2
    

    def process(self, img, low, high):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, low, high)
        mask = cv.medianBlur(mask, 5)
        return mask
       
    def put_text(self, img, mask, color):
        r = []
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        for i in contours:
            area = cv.contourArea(i)
            cv.drawContours(img, i, -1, (255, 0, 0), 2)
            if area > 300 and area < 250000:
                p = cv.arcLength(i, True)
                approx = cv.approxPolyDP(i, 0.02 * p, True)
                x, y, w, h = cv.boundingRect(approx)
                cv.putText(img, color, (int(x + (w // 2)), int(y + (h // 3))), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                r.append([x + (w // 2),y + (h // 2)])
        return r

    def detect_color(self, img):
        r_red = self.process(img, self.red_low, self.red_high)
        r_blue = self.process(img, self.blue_low, self.blue_high)
        r_green = self.process(img, self.green_low, self.green_high)
        r_yellow = self.process(img, self.yellow_low, self.yellow_high)
        r_result=self.put_text(img, r_red, "red/orange")
        b_result=self.put_text(img, r_blue, "blue")
        g_result=self.put_text(img, r_green, "green")
        y_result=self.put_text(img, r_yellow, "yellow")
        return r_result, b_result, g_result, y_result