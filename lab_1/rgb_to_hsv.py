import numpy as np
import cv2
kitten = cv2.imread("/home/temporary/Pictures/kitten.jpg", 1)
kitten = cv2.resize(kitten, (480,340))
cv2.imshow("Image",kitten)
cv2.moveWindow("Image",0,-20)

hsv = cv2.cvtColor(kitten, cv2.COLOR_BGR2HSV)
hsv = cv2.resize(hsv, (480,340))
cv2.imshow("HSV",hsv)
cv2.moveWindow("HSV",600,20)
cv2.waitKey(0)
cv2.destroyAllWindows()
