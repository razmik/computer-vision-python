"""
Source: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

Dense Optical Flow in OpenCV

This computes the optical flow for all the points in the frame.
It is based on Gunner Farneback algorithm which is explained in - Two-Frame Motion Estimation Based on Polynomial
Expansion - by Gunner Farneback in 2003.

Below sample shows how to find the dense optical flow using above algorithm.
We get a 2-channel array with optical flow vectors, (u,v). We find their magnitude and direction.
We color code the result for better visualization. Direction corresponds to Hue value of the image.
Magnitude corresponds to Value plane.
"""

import cv2, sys
import numpy as np

category = "boxing"
filename = "person04_boxing_d1_uncomp"
# filepath = "../../inputdata/" + category + "/" + filename + ".avi"
filepath = "C:/Users\pc\Downloads\Datasets\cctv\dideoplayback.mp4".replace('\\', '/')

cap = cv2.VideoCapture(filepath)

ret, frame1 = cap.read()
prevs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])
    # http://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prevs, next, 0.5, 0.5, 3, 15, 3, 5, 1.2, 0)
    # print(flow.shape)
    # sys.exit(0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('original', frame2)
    cv2.imshow('optical_flow', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()
