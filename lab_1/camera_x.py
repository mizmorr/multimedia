import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
 print("Cannot open camera")
 exit()
while True:
 ret, frame = cap.read()
 dim = frame.shape
 height, width =dim[0],dim[1]

 if not ret:
    break
 center = (int(2*width/4),int(2*height/4))
 gray = frame
 cv.rectangle(gray,(center[0]-90,center[1]-10),(center[0]+90,center[1]+10), (0, 0, 300), 2,8)
 cv.rectangle(gray,(center[0]-10,center[1]+10),(center[0]+10,center[1]+90), (0, 0, 300) , 2,8)
 cv.rectangle(gray,(center[0]-10,center[1]-10),(center[0]+10,center[1]-90), (0, 0, 300) , 2,8)

 cv.imshow('frame', gray)
 if cv.waitKey(1) == ord('q'):
    break

cap.release()
cv.destroyAllWindows()
