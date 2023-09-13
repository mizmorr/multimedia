import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
  print('bad')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))

while(True):
  ret, frame = cap.read()

  if ret == True:

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out.write(frame)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  else:
    break

cap.release()
out.release()

cv2.destroyAllWindows()
