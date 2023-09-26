import requests
import cv2
import numpy as np

url = "http://10.121.14.68:8080/shot.jpg"


while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = cv2.resize(img, (720,480))
    cv2.imshow("Android_cam", img)
    cv2.moveWindow("Android_cam",500,0)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
