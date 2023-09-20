import cv2
import numpy as np

window_capture_name = 'original'
window_detection_name = 'detected'



capture = cv2.VideoCapture("/home/temporary/Videos/circle.mp4")

kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = capture.read()

    if not ret:
        break

    cv2.namedWindow(window_capture_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_detection_name, cv2.WINDOW_NORMAL)

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (0, 180, 80), (10, 255, 255))
    frame_threshold2 = cv2.inRange(frame_HSV, (170, 180, 80), (180, 255, 255))
    combine = frame_threshold | frame_threshold2

    opening = cv2.morphologyEx(combine, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


    moments = cv2.moments(closing, 1)
    m00 = moments['m00']
    m10 = moments['m10']
    m01 = moments['m01']

    if m00 > 3000:
        x = int(m10/m00)
        y = int(m01/m00)
        cv2.rectangle(frame, (x - 300, y + 300), (x + 300, y - 300), (255,200,0), 3)
        cv2.rectangle(closing, (x - 5, y + 5), (x + 5, y - 5), (0,0,0), 3)


    cv2.imshow(window_capture_name, frame)
    cv2.imshow(window_detection_name, closing)
    cv2.moveWindow("detected",500,80)

    if cv2.waitKey(1) & 0xFF == 27:
        break
