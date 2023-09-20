import cv2
import numpy as np
# read the image
img = cv2.imread("/home/temporary/Pictures/dog.jpeg", 0)
img = cv2.resize(img, (480,340))

# binarize the image
binr = cv2.threshold(img, 0, 255,
                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# Бинаризация изображения.
# Мы определяем ядро ​​3×3, заполненное единицами

# define the kernel
kernel = np.ones((3, 3), np.uint8)

# opening the image
opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN,
                           kernel, iterations=1)
closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)

opening = cv2.resize(opening, (480,340))
cv2.imshow("opening",opening)
cv2.imshow("original",img)
cv2.imshow("closing",closing)
cv2.moveWindow("original",600,80)
cv2.moveWindow("closing",300,400)
cv2.waitKey(0)
cv2.destroyAllWindows()
