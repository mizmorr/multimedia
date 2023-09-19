from __future__ import print_function
import cv2 as cv
import numpy as np

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'


cap = cv.VideoCapture(0)

while True:

 ret, frame = cap.read()
 if frame is None:
    break
 frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 lower1 = np.array([0, 100, 20])
 upper1 = np.array([10, 255, 255])

 lower2 = np.array([160,100,20])
 upper2 = np.array([179,255,255])
 lower_mask = cv.inRange(frame_HSV, lower1, upper1)
 upper_mask = cv.inRange(frame_HSV, lower2, upper2)
 full_mask = lower_mask | upper_mask

 cv.imshow(window_capture_name, frame)
 cv.moveWindow('Object Detection',700,55)
 cv.imshow(window_detection_name, full_mask)

 key = cv.waitKey(30)
 if key == ord('q') or key == 27:
    break
