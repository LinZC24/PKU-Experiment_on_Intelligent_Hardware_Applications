import numpy as np
import cv2 as cv
from picamera2 import Picamera2
import time

from goto import with_goto
from serial.tools import list_ports
from pydobot import Dobot
from arm import Arm
from detect import vision


cam = Picamera2()
cam.still_configuration.main.size = (640, 480)
cam.still_configuration.main.format = 'RGB888'
cam.configure("still")
cam.start()
time.sleep(1)

ix=180
iy=0
iz=0
ir=0
h=iz
rz=iz

original_points = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
target_points = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])

v = vision()
p = list_ports.comports()[0].device
print(p)
d = Dobot(port=p, verbose=True)
print("init position\n")
d.move_to(ix,iy,iz,ir)
print("start\n")
print("init the destination:\n")

x0=0
y0=0
z0=0
gnd=-63
xc=x0
yc=y0
zc=z0
h=z0
i=0
m=False

while True:
    x0 = int(input("x:\n"))
    if x0 >= 180 and x0 <= 300:
        break
    print("invalid position, try again!")
while True:
    y0 = int(input("y:\n"))
    if y0>=-150 and y0 <= 150:
        break
    print("invalid position, try again!")
while True:
    z0=int(input("block height:\n"))
    zc=gnd+z0
    h=gnd+z0
    break
    #print("invalid position, try again!")

# img = cam.capture_array("main")
# v.detect_color(img)
# cv.imshow("result", img)
# cv.waitKey(0)
# 
# def show():
#     img = cam.capture_array("main")
#     [r,g,b,y]=v.detect_color(img)
#     print(r,g,b,y)
#     cv.imshow("result", img)
#     cv.waitKey(4)
# def action():
#     input("select a color:\n")
#     return
#
def move():
    d.move_to(xc, yc, h, ir,wait=True)
    d.suck(True)
    d.move_to(xc, yc, 50, ir,wait=True)
    d.move_to(x0,y0,zc,ir,wait=True)
    d.suck(False)
    d.move_to(ix,iy,rz,ir,wait=True)
    print(xc,yc)
    (x1, y1, z1, r1, j1, j2, j3, j4) = d.pose()
    print(f'x:{x1}, y:{y1}, z:{z1}, r:{r1}')
    global m
    m=False
    return
    
def trans(xc, yc):
    x=(xc*120)//640+180
    y=((yc-240)*(-200))//(480)
    return x,y
    
def mouse(event, x, y, flags, param):
    global i
    if event == cv.EVENT_LBUTTONDOWN:
        if i == 4:
            perspective_matrix = cv.getPerspectiveTransform(original_points, target_points)
            global output
            output = cv.warpPerspective(img, perspective_matrix, (640,480))
            #v.detect_color(output)
            #cv.imshow('Output', output)
            #cv.waitKey(0)
            return
        print(x, y)
        original_points[i] = [x, y]
        i = i + 1
        

print("preview and prepare")
while True:
    p = cam.capture_array("main")
    cv.imshow("preview", p)
    
    if cv.waitKey(1) == ord('q'):
        cv.destroyWindow('preview')
        break
while True:
    img = cam.capture_array("main")
    output=img
    #v.detect_color(img)
    cv.imshow("result", img)
    cv.setMouseCallback('result', mouse)
    cv.waitKey(0)
    cv.destroyWindow('result')
#     (x0, y0) = v.detect_edge(img)
#     print(f'x:{x0}, y:{y0}')
#     while True:
#         [r,g,b,y]=v.detect_color(img)
#         print(r,g,b,y)
#         cv.imshow("result", img)
#         cv.waitKey(5)
#     if cv.waitKey(1) == ord("c"):
#         continue 
    #while True:
#     act = input("input an action")
#     if act == "move":
#         (x0, y0, z0, r0, j1, j2, j3, j4) = d.pose()
#         x = input("input x position\n")
#         y = input("input y position\n")
#         z = input("input z position\n")
#         r = input("input r position\n")
#         x = float(x)
#         y = float(y)
#         z = float(z)
#         r = float(r)
#         d.move_to(x, y, z, r=r0, wait = True)
#         (x1, y1, z1, r1, j1, j2, j3, j4) = d.pose()
#         print(f'x:{x1}, y:{y1}, z:{z1}, r:{r1}')
#     else:
#         break
    #while True:
    [r,b,g,y]=v.detect_color(output)
    print(r,g,b,y)
    cv.imshow("preview",output)
#         cv.destroyWindow('preview')
    cv.waitKey(0)
    cv.destroyWindow('preview')
        #print(r)
    color = input("select a color\n")
    if color == "red" and len(r) != 0:
        print("you selected red\n")
        xc=r[0][0]
        yc=r[0][1]
        m=True
    elif color == "orange" and len(r) != 0:
        print("you selected red\n")
        xc=r[0][0]
        yc=r[0][1]
        m=True
    elif color == "green" and len(g) != 0:
        print("you selected green\n")
        xc=g[0][0]
        yc=g[0][1]
        m=True
    elif color == "blue" and len(b) != 0:
        print("you selected blue\n")
        xc=b[0][0]
        yc=b[0][1]
        m=True
    elif color == "yellow" and len(y) != 0:
        print("you selected yellow\n")
        xc=y[0][0]
        yc=y[0][1]
        m=True
    else:
        print("invalid color!\n")
#        xc=int(input("input xc"))
#         yc=int(input("input yc"))
        
    if m:
        xc, yc = trans(xc, yc)
        move()
        zc=zc+z0
        #time.sleep(5)
#         if cv.waitKey(1) == ord("q"):
#             break
# 
# while True:
#     show()
#     action()
cv.destroyAllWindows()
d.close()
cam.stop()
