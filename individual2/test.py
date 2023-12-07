import cv2
SAMPLE_IMAGE = "/home/temporary/Pictures/kittens/kitten2.jpg"
image = cv2.imread(SAMPLE_IMAGE,cv2.IMREAD_UNCHANGED)
image = cv2.resize(image,(800,600))
blurred = cv2.GaussianBlur(image,(5,5),3)
blurred2 = cv2.GaussianBlur(image,(5,5),4)

cv2.imshow('dsa',blurred)
cv2.imshow('dsa2',blurred2)

cv2.waitKey(0)
cv2.destroyAllWindows()
