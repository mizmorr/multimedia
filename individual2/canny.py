import os
import cv2
import math
import numpy as np
from scipy import ndimage
from common import *
LOW_THRESHOLD_RATIO = 0.09
HIGH_THRESHOLD_RATIO = 0.17
WEAK_PIXEL = 100
STRONG_PIXEL = 255

SAMPLE_IMAGE = "/home/temporary/Pictures/kittens/kitten1.jpeg"



def sobel(im):
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    out2 = cv2.filter2D(im, cv2.CV_64F, Gx, None, (-1,-1), 0, cv2.BORDER_DEFAULT)
    out3 = cv2.filter2D(im, cv2.CV_64F, Gy, None, (-1,-1), 0, cv2.BORDER_DEFAULT)
    out_result = np.hypot(out2, out3)
    out_result = cv2.convertScaleAbs(out_result.copy())
    matr_grd_dir = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            matr_grd_dir[i,j] = get_angle(out2[i,j],out3[i,j])

    return out_result,matr_grd_dir

def non_max(img,D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M):
        for j in range(1, N):
            try:
                q = 255
                r = 255

               # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    return Z


def non_max_sup(matr_grd_length,matr_grd_dir):
    M, N = matr_grd_dir.shape
    border = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            grad = matr_grd_length[i,j]
            direct = matr_grd_dir[i,j]
            if i==0 or i == M-1 or j==0 or j == N-1:
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
    return border

def threshold(img):
    hiThresh = img.max()*HIGH_THRESHOLD_RATIO
    loThresh = hiThresh * LOW_THRESHOLD_RATIO
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.float32)

    strong_i, strong_j = np.where(img >= hiThresh)

    weak_i, weak_j = np.where((img >= loThresh) & (img <= hiThresh))

    res[strong_i, strong_j] = STRONG_PIXEL
    res[weak_i, weak_j] = WEAK_PIXEL

    return res

def hysteresis(img):
    M, N = img.shape
    res = np.copy(img)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == WEAK_PIXEL):
                try:
                    if (img[i+1, j-1] == STRONG_PIXEL) or (img[i+1, j] == STRONG_PIXEL) or (img[i+1, j+1] == STRONG_PIXEL) or (img[i, j-1] == STRONG_PIXEL) or (img[i, j+1] == STRONG_PIXEL) or (img[i-1, j-1] == STRONG_PIXEL) or (img[i-1, j] == STRONG_PIXEL) or (img[i-1, j+1] == STRONG_PIXEL):
                        res[i, j] = STRONG_PIXEL
                    else:
                        res[i, j] = 0
                except IndexError as e:
                    pass
    return res

image = cv2.imread(SAMPLE_IMAGE,cv2.IMREAD_ANYCOLOR)
image = cv2.resize(image,(600,480))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_image,(5,5),2)

gradientMat, thetaMat,dirrect = sobel_filters(blurred)


gradient, theta,direct = scharr_operator(blurred)


grad, theta = sobel(blurred)

non_m = non_max(gradientMat, thetaMat)
non_max2 = non_max_sup(gradientMat,dirrect)

non3 = non_max(grad,theta)

non_max_sh = non_max_sup(gradient,direct)
# non_max_img = non_max_sup(blurred,gradientMat,thetaMat)

threshed = threshold(non_m)

threshed2 = threshold(non_max2)


thresh4 = threshold(non3)

thresh_sh = threshold(non_max_sh)
# cv2.imshow('rgb', image)


canny1 = hysteresis(threshed)
canny2 = hysteresis(threshed2)
canny4 = hysteresis(thresh4)
cannysh = hysteresis(thresh_sh)
cv2.imshow('grad+thet',canny1)
cv2.imshow('grad+dir',canny2)
cv2.imshow('conv2d',canny4)
cv2.imshow('sharr',cannysh)
# cv2.imshow('blured',blurred)


canny_image_cv2 = cv2.Canny(image, WEAK_PIXEL, STRONG_PIXEL)
cv2.imshow('canny',canny_image_cv2)
# cv2.imshow('canny_handmade',non_max_img)

cv2.waitKey(0)

cv2.destroyAllWindows()
