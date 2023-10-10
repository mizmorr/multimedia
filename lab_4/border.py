import math
import cv2
import numpy as np

def get_angle(x,y):
    tg = y/x if x!=0 else 999
    if y>0:
        if x>0:
            if tg<0.414:
                return 2
            if tg>2.414:
                return 4
            else:
                return 3
        else:
            if tg<-2.414:
                return 4
            if tg<-0.414:
                return 5
            elif tg>-0.414:
                return 6
    else:
        if x>0:
            if tg<-2.414:
                return 0
            if tg<-0.414:
                return 1
            if tg>-0.414:
                return 2
        else:
            if tg>2.414:
                return 0
            if tg<2.414:
                return 7
            if tg < 0.414:
                return 6




img = cv2.imread('/home/temporary/Pictures/figures.png',cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (450,300))
def roll(img, kernel):
    size = len(kernel)
    s = size // 2
    matr = img.copy()
    for i in range(s, len(matr)-s):
        for j in range(s, len(matr[i])-s):
            val = 0
            for k in range(-s, s+1):
                for l in range(-s, s+1):
                    val += img[i + k][j + l] * kernel[k+s][l + s]
            matr[i][j] = val

    return matr

Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

img_Gx = roll(img, Gx)
img_Gy = roll(img,Gy)
matr_grd_length = np.zeros(img.shape)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        matr_grd_length[i,j] = math.sqrt(img_Gx[i,j]**2 + img_Gy[i,j]**2)

matr_grd_dir = np.zeros(img.shape)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        matr_grd_dir[i,j] = get_angle(img_Gx[i,j],img_Gy[i,j])


border  = img.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        grad = matr_grd_length[i,j]
        direct = matr_grd_dir[i,j]
        if i==0 or i == img.shape[0]-1 or j==0 or j == img.shape[1]-1:
            border[i,j] = 0
        else:
                if (direct == 0 or direct == 4):
                    x_shift = 0
                elif (direct > 0 and direct < 4):
                    x_shift = 1
                else:
                    x_shift = -1

                if (direct == 2 or direct == 6):
                    y_shift = 0
                elif (direct > 2 and direct < 6):
                    y_shift = -1
                else:
                    y_shift = 1

                if grad >= matr_grd_length[i+y_shift][j +
                                                              x_shift] and grad >= matr_grd_length[i-y_shift][j-x_shift]:
                    border[i][j] = 255
                else:
                    border[i][j] = 0

low_level = np.max(matr_grd_length)//25
upper_level = np.max(matr_grd_length)- np.max(matr_grd_length)//25
post_bordered = np.zeros(img.shape)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        grad = matr_grd_length[i][j]
        if border[i][j] == 255:
            if grad > upper_level:
                post_bordered[i,j]=255
            elif grad >= low_level and grad  <= upper_level:
                is_border = False
                for k in range(-1,2):
                    for l in range(-1,2):
                        if is_border:
                            break
                        if border[i+k][j+l] == 255 and matr_grd_length[i+k][j+l] >= low_level:
                            is_border = True
                            break
                    if is_border:
                        post_bordered[i,j] = 255

cv2.imshow('border', border)
cv2.imshow('bordered', post_bordered)
cv2.waitKey(0)
cv2.destroyAllWindows()



