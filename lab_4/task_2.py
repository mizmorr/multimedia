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

def roll(img,kernel):
    size = len(kernel)
    copy = img.copy()
    s = size//2
    for i in range(s,copy.shape[0]-s):
        for j in range(s,copy.shape[1]-s):
            val = 0
            for k in range(-s,s+1):
                for l in range(-s,s+1):
                    val+=copy[i+k,j+l]*kernel[s+k][s+l]
            copy[i,j] = val
    return copy


img = cv2.imread('/home/temporary/Pictures/kitten2.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (350,200))


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

print(matr_grd_dir)
