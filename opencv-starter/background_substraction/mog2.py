"""
Source: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html

BackgroundSubtractorMOG

It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
It uses a method to model each background pixel by a mixture of K Gaussian distributions (K = 3 to 5).
The weights of the mixture represent the time proportions that those colours stay in the scene.
The probable background colours are the ones which stay longer and more static.
One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel.
It provides better adaptibility to varying scenes due illumination changes etc.

Here, you have an option of selecting whether shadow to be detected or not.
If detectShadows = True (which is so by default), it detects and marks shadows,
but decreases the speed. Shadows will be marked in gray color.
"""

import numpy as np
import cv2

category = "boxing"
filename = "person04_boxing_d1_uncomp"
# filepath = "../../inputdata/" + category + "/" + filename + ".avi"
filepath = "C:/Users\pc\Downloads\Datasets\cctv\dideoplayback.mp4".replace('\\', '/')

cap = cv2.VideoCapture(filepath)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# fgbg_noshadow = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while True:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    # fgmask_noshadow = fgbg_noshadow.apply(frame)

    # print(frame.shape)
    # print(fgmask.shape)

    cv2.imshow('Frame1_with_shadow',fgmask)
    # cv2.imshow('Frame1_No_shadow',fgmask_noshadow)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
