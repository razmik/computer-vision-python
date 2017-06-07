"""
Source: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html

BackgroundSubtractorKNN
K-nearest neigbours - based Background/Foreground Segmentation Algorithm.


"""

import numpy as np
import cv2

category = "boxing"
filename = "person04_boxing_d1_uncomp"
# filepath = "../../inputdata/" + category + "/" + filename + ".avi"
filepath = "C:/Users\pc\Downloads\Datasets\cctv\dideoplayback.mp4".replace('\\', '/')

cap = cv2.VideoCapture(filepath)

fgbg = cv2.createBackgroundSubtractorKNN()
fgbg_history = cv2.createBackgroundSubtractorKNN(history=10)

while True:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask_history = fgbg_history.apply(frame)

    cv2.imshow('Frame1', fgmask)
    cv2.imshow('Frame1_with_history', fgmask_history)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
