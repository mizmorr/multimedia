from canny import *
from log import *
import cv2

SAMPLE_IMAGE = "/home/temporary/Pictures/kittens/kitten2.jpg"
image = cv2.imread(SAMPLE_IMAGE,cv2.IMREAD_UNCHANGED)
image = cv2.resize(image,(525,350))

kernel_size = 5
delta = 4

image = cv2.GaussianBlur(image,(kernel_size, kernel_size),delta)
cv2.imshow('original',image)


while cv2.waitKey(0)!=ord('e'):

    if cv2.waitKey(0) == ord('o'):
        cv2.destroyAllWindows()
        cv2.imshow('orig',image)

    if cv2.waitKey(0) == ord('s'):
        print('sobel start')
        cv2.destroyAllWindows()
        start = time.time()
        handm = sobel(image,0,kernel_size,delta)
        end = time.time()
        print('first sobel done, time - ', end-start,'s')

        start = time.time()
        thet = sobel(image,1,kernel_size,delta)
        end = time.time()
        print('second sobel done, time - ', end-start,'s')

        start = time.time()
        direct = sobel(image,2,kernel_size,delta)
        end = time.time()
        print('third sobel done, time - ', end-start,'s')

        start = time.time()
        canny_opencv = sobel(image,3,kernel_size,delta)
        end = time.time()
        print('fourth sobel done, time - ', end-start,'s')
        print('lets display')
        cv2.imshow('first',handm)
        cv2.imshow('second',thet)
        cv2.imshow('third',direct)
        cv2.imshow('fourth',canny_opencv)
        cv2.moveWindow("second",1000,0)
        cv2.moveWindow("third",0,600)
        cv2.moveWindow("fourth",1000,600)

    if cv2.waitKey(0) == ord('n'):
        print('not sobel start')
        cv2.destroyAllWindows()
        start = time.time()
        prewitt = not_sobel(image,1,kernel_size,delta)
        end = time.time()
        print('prewitt done, time - ', end-start,'s')
        start = time.time()
        scharr = not_sobel(image,0,kernel_size,delta)
        end = time.time()
        print('scharr done, time - ', end-start,'s')
        cv2.imshow('prewitt',prewitt)
        cv2.imshow('scharr',scharr)
        cv2.moveWindow("scharr",600,0)

    if cv2.waitKey(0) == ord('l'):
        cv2.destroyAllWindows()
        start = time.time()
        log = gaussian(image,1,175,kernel_size)
        end = time.time()
        print('Laplasian of Gaussian done, time - ', end-start,'s')
        cv2.imshow('Laplasian of Gaussian',log)
        cv2.moveWindow('Laplasian of Gaussian',1000,0)

    if cv2.waitKey() == ord('d'):
        cv2.destroyAllWindows()
        start = time.time()
        dog = gaussian(image,0,175,kernel_size)
        end = time.time()
        print('Difference of Gaussian done, time - ', end-start,'s')
        cv2.imshow('Difference of Gaussian',dog)
        cv2.moveWindow('Difference of Gaussian',1000,0)

print('exit....')
cv2.waitKey(0)
cv2.destroyAllWindows()

