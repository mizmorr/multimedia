import cv2

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'



capture = cv2.VideoCapture("/home/temporary/Videos/circle.mp4")


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
while True:
    ret, frame = capture.read()

    if not ret:
        break

    cv2.namedWindow(window_capture_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_detection_name, cv2.WINDOW_NORMAL)

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (0, 180, 80), (10, 255, 255))
    frame_threshold2 = cv2.inRange(frame_HSV, (170, 180, 80), (180, 255, 255))
    combine = frame_threshold + frame_threshold2

    opening = cv2.morphologyEx(combine, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


    moments = cv2.moments(closing, 1)
    dm00 = moments['m00']
    dm10 = moments['m10']
    dm01 = moments['m01']

    if dm00 > 2500:
        x = int(dm10/dm00)
        y = int(dm01/dm00)
        cv2.rectangle(frame, (x - 300, y + 300), (x + 300, y - 300), (255,200,0), 3)
        cv2.rectangle(closing, (x - 5, y + 5), (x + 5, y - 5), (0,0,0), 3)


    cv2.imshow(window_capture_name, frame)
    cv2.imshow(window_detection_name, closing)

    if cv2.waitKey(1) & 0xFF == 27:
        break
