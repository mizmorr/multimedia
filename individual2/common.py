import numpy as np
import cv2
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

def convolve(img, kernel):
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

def sobel_filters(img):
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]], dtype=np.float32)

    return filter(img,Kx,Ky)

def filter(img,Kx,Ky):
    Ix = convolve(np.float32(img), Kx)
    Iy = convolve(np.float32(img), Ky)
    G = np.hypot(Ix, Iy)
    G = np.array(G,dtype=np.float32)
    G = cv2.convertScaleAbs(G.copy())
    matr_grd_dir = np.zeros(img.shape)
    theta = np.arctan2(Iy, Ix)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr_grd_dir[i,j] = get_angle(Ix[i,j],Iy[i,j])

    return G, theta,matr_grd_dir

def filter2(img,Kx,Ky):
    Ix = cv2.filter2D(src=img, ddepth=-1, kernel=Kx)
    Iy = cv2.filter2D(src=img, ddepth=-1, kernel=Ky)
    G = np.hypot(Ix, Iy)
    G = np.array(G,dtype=np.float32)
    G = cv2.convertScaleAbs(G.copy())
    matr_grd_dir = np.zeros(img.shape)
    theta = np.arctan2(Iy, Ix)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            matr_grd_dir[i,j] = get_angle(Ix[i,j],Iy[i,j])

    return G, theta,matr_grd_dir

def scharr_operator(img):
    SCx = np.array([
        [3, 0,-3],
        [10, 0, -10],
        [3, 0, -3]], dtype=np.float32)
    SCy = np.array([
        [3, 10,3],
        [0, 0, 0],
        [-3, -10, -3]], dtype=np.float32)
    return filter2(img,SCx,SCy)


