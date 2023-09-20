import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
 print("Cannot open camera")
 exit()
while True:
 ret, frame = cap.read()
 hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 cv.imshow('rgb', frame)
 cv.moveWindow("hsv",700,85)
 cv.imshow('hsv',hsv)
 if cv.waitKey(1) == ord('q'):
    break

cap.release()
cv.destroyAllWindows()
