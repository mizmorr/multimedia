import cv2 as cv

cap = cv.VideoCapture('/home/temporary/Videos/test.mp4',cv.CAP_ANY)
# cap = cv.VideoCapture('/home/temporary/Videos/test.mp4',cv.CAP_PROP_FPS)


while(True):
    ret, frame = cap.read()
    # frame = cv.resize(frame, (224, 224))
    frame = cv.resize(frame, (480, 250))
    frame2 = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)

    if not(ret):
        break
    cv.imshow('frame',frame)
    cv.imshow('frame gray',frame2)
    cv.moveWindow("frame gray",600,-20)

    if cv.waitKey(1) & 0xFF == 27:
        break
