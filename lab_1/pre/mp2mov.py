import cv2
def readIPWriteTOFile():
    video = cv2.VideoCapture('/home/temporary/Videos/test.mp4',cv2.CAP_ANY)
    ok, img = video.read()
    img = cv2.resize(img, (480, 250))

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w, h))
    while (True):
        ok, img = video.read()
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

readIPWriteTOFile()
