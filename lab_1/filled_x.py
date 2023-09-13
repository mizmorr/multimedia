import cv2 as cv
# img1 = cv.imread('/home/temporary/Pictures/night.png',cv.IMREAD_GRAYSCALE)
img = cv.imread('/home/temporary/Pictures/kitten.jpg',cv.IMREAD_ANYCOLOR)
img = cv.resize(img, (480, 320))
shape = img.shape

height,width = shape[0],shape[1]

color = img[int(width/2), int(height/2)]
print(height,width)
print(color[0])
new_color,max = [255,0,0], color[0]
if max<color[1] and max<color[2]:
    new_color = [0,0,255]
elif color[1]>color[2] and max<color[1]:
    new_color = [0,255,0]

cv.rectangle(img,(int(width/2)-60,int(height/2)-6),(int(width/2)+60,int(height/2)+6),new_color , -1)
cv.rectangle(img,(int(width/2)-6,int(height/2)-60),(int(width/2)+6,int(height/2)+60),new_color , -1)

cv.imshow('Kitten2', img)

cv.waitKey(0)

cv.destroyAllWindows()
