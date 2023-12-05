
# import numpy as np
# import os
# import cv2

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


# def sobel_filters(img):
#     Kx = np.array([
#         [-1, 0, 1],
#         [-2, 0, 2],
#         [-1, 0, 1]], dtype=np.float32)
#     Ky = np.array([
#         [1, 2, 1],
#         [0, 0, 0],
#         [-1, -2, -1]], dtype=np.float32)
#     Ix = convolve(img, Kx)
#     Iy = convolve(img, Ky)
#     d = np.hypot(Ix,Iy)
#     d *= 255.0 / d.max()
#     return d

# def test_sobel(img):
#     grayscale_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
#     Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
#     [rows, columns] = np.shape(grayscale_image)
#     sobel_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

#     for i in range(rows - 2):
#         for j in range(columns - 2):
#             gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
#             gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
#             sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
#     return sobel_filtered_image



# def test_dob(img):
#     Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
#     Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
#     gx = convolution(img,Gx)
#     gy = convolution(img,Gy)
#     d = np.hypot(gx,gy)
#     print(d[0])

# def Canny_detector(img, weak_th = None, strong_th = None):

#     # conversion of image to grayscale
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Noise reduction step
#     img = cv2.GaussianBlur(img, (5, 5), 1.4)

#     # Calculating the gradients
#     gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 5)
#     gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 5)
#     # Ix,Iy = sobel_filters(img)
#     # G = np.hypot(Ix, Iy)
#     # print(G)
#     # print()
#     G = np.hypot(gx, gy)

#     print(G[0])
#     # Conversion of Cartesian coordinates to polar
#     # mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)

#     # # setting the minimum and maximum thresholds
#     # # for double thresholding
#     # mag_max = np.max(mag)
#     # if not weak_th:weak_th = mag_max * 0.1
#     # if not strong_th:strong_th = mag_max * 0.5

#     # # getting the dimensions of the input image
#     # height, width = img.shape

#     # # Looping through every pixel of the grayscale
#     # # image
#     # for i_x in range(width):
#     #     for i_y in range(height):

#     #         grad_ang = ang[i_y, i_x]
#     #         grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)

#     #         # selecting the neighbours of the target pixel
#     #         # according to the gradient direction
#     #         # In the x axis direction
#     #         if grad_ang<= 22.5:
#     #             neighb_1_x, neighb_1_y = i_x-1, i_y
#     #             neighb_2_x, neighb_2_y = i_x + 1, i_y

#     #         # top right (diagonal-1) direction
#     #         elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
#     #             neighb_1_x, neighb_1_y = i_x-1, i_y-1
#     #             neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

#     #         # In y-axis direction
#     #         elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
#     #             neighb_1_x, neighb_1_y = i_x, i_y-1
#     #             neighb_2_x, neighb_2_y = i_x, i_y + 1

#     #         # top left (diagonal-2) direction
#     #         elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
#     #             neighb_1_x, neighb_1_y = i_x-1, i_y + 1
#     #             neighb_2_x, neighb_2_y = i_x + 1, i_y-1

#     #         # Now it restarts the cycle
#     #         elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
#     #             neighb_1_x, neighb_1_y = i_x-1, i_y
#     #             neighb_2_x, neighb_2_y = i_x + 1, i_y

#     #         # Non-maximum suppression step
#     #         if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
#     #             if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
#     #                 mag[i_y, i_x]= 0
#     #                 continue

#     #         if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
#     #             if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
#     #                 mag[i_y, i_x]= 0

#     # weak_ids = np.zeros_like(img)
#     # strong_ids = np.zeros_like(img)
#     # ids = np.zeros_like(img)

#     # # double thresholding step
#     # for i_x in range(width):
#     #     for i_y in range(height):

#     #         grad_mag = mag[i_y, i_x]

#     #         if grad_mag<weak_th:
#     #             mag[i_y, i_x]= 0
#     #         elif strong_th>grad_mag>= weak_th:
#     #             ids[i_y, i_x]= 1
#     #         else:
#     #             ids[i_y, i_x]= 2


#     # # finally returning the magnitude of
#     # # gradients of edges
#     # return mag
# SAMPLE_IMAGE = "/home/temporary/Pictures/kittens/kitten1.jpeg"



# image = cv2.imread(SAMPLE_IMAGE,cv2.IMREAD_ANYCOLOR)
# # image = cv2.resize(image,(600,480))
# # # # calling the designed function for
# # # # finding edges
# image = cv2.resize(image,(600,480))
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray_image,(5,5),2)
# img2 = sobel_filters(blurred)
# canny_img = Canny_detector(image,100,255)
# print(img2[0])
# test_dob(image)

# # cv2.imshow('canny',canny_img)
# # cv2.imshow('canny_handmade',non_max_img)

# # cv2.waitKey(0)

# # cv2.destroyAllWindows()

# # Displaying the input and output image
# # plt.figure()
# # f, plots = plt.subplots(2, 1)
# # plots[0].imshow(frame)
# # plots[1].imshow(canny_img)

import cv2
import numpy as np
im = cv2.imread('/home/temporary/Pictures/kittens/kitten1.jpeg'); #// Save image to computer first
im = cv2.resize(im,(600,480))
gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im = cv2.GaussianBlur(gray_image,(5,5),2)
#// Call using built-in Sobel
out1 = cv2.Sobel(im, cv2.CV_64F, 1, 0, 5)
out_1 = cv2.Sobel(im, cv2.CV_64F,0,1,5)
out_1_r = np.hypot(out1, out_1)
out_1_r = cv2.convertScaleAbs(out_1_r.copy())
#// Create custom kernel
Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
# xVals = np.array([0.125,0,-0.125,0.25,0,-0.25,0.125,0,-0.125]).reshape(3,3)

#// Call filter2D
out2 = cv2.filter2D(im, cv2.CV_64F, Gx, None, (-1,-1), 0, cv2.BORDER_DEFAULT)
out3 = cv2.filter2D(im, cv2.CV_64F, Gy, None, (-1,-1), 0, cv2.BORDER_DEFAULT)
out_result = np.hypot(out2, out3)
out_result = cv2.convertScaleAbs(out_result.copy())

out5 = convolve(np.float32(im),Gx)
out_5 = convolve(np.float32(im),Gy)
out_5_r = np.hypot(out5, out_5)
out_5_r = cv2.convertScaleAbs(out_5_r.copy())

# cv2.imshow('Output 1', out_1_r)
cv2.imshow('Output 2', out_result)
cv2.imshow('Output 3', out_5_r)

cv2.waitKey(0)
cv2.destroyAllWindows()
