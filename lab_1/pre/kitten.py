import cv2 as cv
# img1 = cv.imread('/home/temporary/Pictures/night.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread('/home/temporary/Pictures/kitten.jpg',cv.IMREAD_ANYCOLOR)

img1 = cv.imread('/home/temporary/Pictures/kitten.jpg',cv.IMREAD_GRAYSCALE)
# img1 = cv.imread('/home/temporary/Pictures/kitten.jpg',cv.IMREAD_ANYCOLOR)


# cv.namedWindow('Kitten', cv.WINDOW_NORMAL)
# cv.namedWindow('Kitten', cv.WINDOW_AUTOSIZE)
cv.namedWindow('Kitten', cv.WINDOW_FREERATIO)


cv.imshow('Kitten', img1)
cv.imshow('Kitten2', img2)

cv.waitKey(0)

cv.destroyAllWindows()
