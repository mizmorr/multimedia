import math
import cv2

import numpy as np


def gauss(x,y,sig,a,b):
    e = np.exp(-((x-a)**2+(y-b)**2)/(2*(sig**2)))
    return 1/(2*math.pi*sig**2)*e

def gaussian_blur(img,size,d):
    kernel = np.zeros((size,size))
    a = b = (size+1)/2
    for i in range(size):
        for j in range(size):
            kernel[i,j] = gauss(i,j,d,a,b)
    print(kernel)
    sum = np.sum(kernel)
    for i in range(size):
        for j in range(size):
            kernel[i,j]/=sum
    print(kernel)
    sum = np.sum(kernel)
    print(sum)
    copy = img.copy()
    s = size//2
    for i in range(s,copy.shape[0]-s):
        for j in range(s,copy.shape[1]-s):
            val = 0
            for k in range(-s,s+1):
                for l in range(-s,s+1):
                    val+=img[i+k,j+l]*kernel[s+k,s+l]
            copy[i,j] = val
    return copy

def run():
    img = cv2.imread('/home/temporary/Pictures/kitten2.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (600,600))

    cv2.imshow('original',img)

    blur = gaussian_blur(img,3,2)
    cv2.imshow('made blur',blur)

    blur_opencv = cv2.GaussianBlur(img, (3,3),2)
    cv2.imshow('opencv blur',blur_opencv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


run()
