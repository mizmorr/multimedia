import cv2
bgr_img = cv2.imread('/home/temporary/Pictures/kitten.jpg',cv2.IMREAD_GRAYSCALE)

hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

cv2.imwrite('hsv_image.jpg', hsv_img)


cv2.imshow('HSV image', hsv_img)
cv2.imshow('RGB image', bgr_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
