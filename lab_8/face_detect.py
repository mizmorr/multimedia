# import the necessary packages
import argparse
import time
import cv2
import os

detectorPaths = {
	"face": "haarcascade_frontalface_default.xml",
    "eyes": "haarcascade_eye.xml",
}
# initialize a dictionary to store our haar cascade detectors
print("[INFO] loading haar cascades...")
detectors = {}
# loop over our detector paths
for (name, path) in detectorPaths.items():
	# load the haar cascade from disk and store it in the detectors
	# dictionary
	detectors[name] = cv2.CascadeClassifier(path)

video = cv2.VideoCapture("/home/temporary/Videos/main_video.mov",cv2.CAP_ANY)


_, frame = video.read()


fourcc = cv2.VideoWriter_fourcc(*'mp4v')

while True:
        flag, frame = video.read()
        if not flag:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    # perform face detection using the appropriate haar cascade
        faceRects = detectors["face"].detectMultiScale(
		gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
        for (fX, fY, fW, fH) in faceRects:
            faceROI = gray[fY:fY+ fH, fX:fX + fW]
            # apply eyes detection to the face ROI
            eyeRects = detectors["eyes"].detectMultiScale(
                faceROI, scaleFactor=1.1, minNeighbors=10,
                minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
            for (eX, eY, eW, eH) in eyeRects:
                # draw the eye bounding box
                ptA = (fX + eX, fY + eY)
                ptB = (fX + eX + eW, fY + eY + eH)
                cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                (0, 255, 0), 2)

        cv2.imshow('video',frame)

        if cv2.waitKey(1) == ord('q'):
            break

video.release
cv2.destroyAllWindows()
