import numpy as np
import cv2

filepath = "C:/Users\pc\Downloads\Datasets\cctv\dideoplayback.mp4".replace('\\', '/')

cap = cv2.VideoCapture(filepath)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints = sift.detect(gray, None)

    img = cv2.drawKeypoints(gray, keypoints, frame)

    cv2.imshow('Frame1', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
