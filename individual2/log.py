import cv2
import numpy as np
from common import *
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def plot_input(img, title):
    plt.imshow(img, cmap = 'gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def handle_padding(img1, img2):
    M1, N1 = img1.shape[:2]
    M2, N2 = img2.shape[:2]
    padding_x = int(np.abs(M2 - M1)/2)
    padding_y = int(np.abs(N2 - N1)/2)

    img2 = img2[padding_x:M1+padding_x, padding_y: N1+padding_y]
    return img2

im = cv2.imread('/home/temporary/Pictures/kittens/kitten1.jpeg')
im = cv2.resize(im,(600,480))
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=3)
sobel_first_derivative = cv2.magnitude(sobelx,sobely)

DoG_kernel = [
            [0,   0, -1, -1, -1, 0, 0],
            [0,  -2, -3, -3, -3,-2, 0],
            [-1, -3,  5,  5,  5,-3,-1],
            [-1, -3,  5, 16,  5,-3,-1],
            [-1, -3,  5,  5,  5,-3,-1],
            [0,  -2, -3, -3, -3,-2, 0],
            [0,   0, -1, -1, -1, 0, 0]
        ]
LoG_kernel = np.array([
                        [0, 0,  1, 0, 0],
                        [0, 1,  2, 1, 0],
                        [1, 2,-16, 2, 1],
                        [0, 1,  2, 1, 0],
                        [0, 0,  1, 0, 0]
                    ])
def zero_cross_detection(image):
    z_c_image = np.zeros(image.shape)

    for i in range(0,image.shape[0]-1):
        for j in range(0,image.shape[1]-1):
            if image[i][j]>0:
                if image[i+1][j] < 0 or image[i+1][j+1] < 0 or image[i][j+1] < 0:
                    z_c_image[i,j] = 1
            elif image[i][j] < 0:
                if image[i+1][j] > 0 or image[i+1][j+1] > 0 or image[i][j+1] > 0:
                    z_c_image[i,j] = 1
    return z_c_image

def bitwise_and(img1, img2):
    result = np.zeros(img1.shape)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if int(img1[i][j])&int(img2[i][j]):
                result[i][j]=255
    return result

def dog(im,thresh):
    dog_img = convolve2d(im,DoG_kernel)
    # dog_img = cv2.filter2D(src=im, ddepth=-1, kernel=np.float32(DoG_kernel))
    zero = zero_cross_detection(dog_img)
    zero = handle_padding(im, zero)
    cv2.imshow('zero',zero)

    sobel_test = np.empty_like(sobel_first_derivative)

    sobel_test[:] = sobel_first_derivative

    sobel_test[sobel_test > thresh] = 255
    sobel_test[sobel_test < thresh] = 0
    # sobel_test = sobel_test[:,:,0]
    cv2.imshow('boosted',sobel_test)
    # ar = cv2.bitwise_and(np.uint8(zero),np.uint8(sobel_test))
    result = bitwise_and(zero, sobel_test)
    return result

def log(im,thresh):
    log_img = convolve2d(im,LoG_kernel)
    # dog_img = cv2.filter2D(src=im, ddepth=-1, kernel=np.float32(DoG_kernel))
    zero = zero_cross_detection(log_img)
    zero = handle_padding(im, zero)
    cv2.imshow('zero',zero)

    sobel_test = np.empty_like(sobel_first_derivative)

    sobel_test[:] = sobel_first_derivative

    sobel_test[sobel_test > thresh] = 255
    sobel_test[sobel_test < thresh] = 0
    # sobel_test = sobel_test[:,:,0]
    cv2.imshow('boosted',sobel_test)
    # ar = cv2.bitwise_and(np.uint8(zero),np.uint8(sobel_test))
    result = bitwise_and(zero, sobel_test)
    return result

result = dog(im,175)
log = log(im,175)
# print(np.where(result!=0))
cv2.imshow('result',result)
cv2.imshow('log',log)

cv2.waitKey(0)
cv2.destroyAllWindows()

