import sys
import time
import cv2

tracker_types = ['CSRT','MEDIANFLOW','MOSSE']
tracker_type = tracker_types[2]

if tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()

map_deers = {0:'deer.mp4',1:'deer_2.mp4',2:'deer_3.mp4',3:'deer_4.MP4',4:'deer_5.mp4'}
deer_num = int(sys.argv[1])
current_deer = map_deers[deer_num]

video = cv2.VideoCapture("/home/temporary/Videos/"+current_deer)
ret, frame = video.read()
frame = cv2.resize(frame,(1250,900))

frame_height, frame_width = frame.shape[:2]
frame = cv2.resize(frame, [frame_width//2, frame_height//2])

output = cv2.VideoWriter('./result_video/'+current_deer[:-4]+f'_{tracker_type}.avi',
                         cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                         (frame_width//2, frame_height//2), True)
if not ret:
    print('cannot read the video')

bbox = cv2.selectROI(frame, False)
ret = tracker.init(frame, bbox)
# Start tracking
start_time = time.time()

while True:
    ret, frame = video.read()

    if not ret:
        print('something went wrong')
        break
    frame = cv2.resize(frame, [frame_width//2, frame_height//2])
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100,80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75,(0,0,255),2)
    cv2.putText(frame, tracker_type + " Tracker", (100,20),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,170,50),2)
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100,50),
    #             cv2.FONT_HERSHEY_DUPLEX, 0.75, (50,170,50),2)
    cv2.imshow("Tracking", frame)
    output.write(frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

end_time = time.time()
if video.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время работы метода {tracker_type}: {end_time - start_time:.5f} секунд")
    print(f"Частота потери изображения: {1 / ((end_time - start_time) / video.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадр/с")

video.release()
output.release()
cv2.destroyAllWindows()
