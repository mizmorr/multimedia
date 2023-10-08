import cv2



def run():
    img_original =  cv2.imread('/home/temporary/Pictures/kitten2.jpg',cv2.IMREAD_ANYCOLOR)
    img = cv2.imread('/home/temporary/Pictures/kitten2.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (450,300))
    img_original = cv2.resize(img_original, (450,300))
    cv2.imshow('original',img_original)

    cv2.imshow('grey',img)

    blur_opencv = cv2.GaussianBlur(img, (3,3),2)
    cv2.imshow('opencv blur',blur_opencv)
    cv2.moveWindow('grey',600,100)
    cv2.moveWindow('opencv blur',300,400)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


run()
