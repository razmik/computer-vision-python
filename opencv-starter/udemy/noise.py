import numpy as np
import os
from skimage.util import random_noise
import matplotlib.pyplot as plt
import scipy.misc
import cv2

"""
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
"""


def noisy(image, noise_typ='gauss', var=0.1):
    if noise_typ == "gauss":
        if len(image.shape) == 2:
            row, col = image.shape
            mean = 0
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)
            noisy = image + gauss
        else:
            row, col, ch = image.shape
            mean = 0
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
    return noisy

"""
Noising image
"""
# filepath = "..\data\group.jpg".replace('\\', '/')
#
# img = cv2.imread(filepath)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# noised_img = noisy(gray, var=1000)
#
# # plt.imshow(gray)
# # plt.show()
#
# # noised_img = random_noise(img) # skimage - add noise function
#
# scipy.misc.imsave('outfile.jpg', noised_img)

"""
Denoising
"""
filepath = 'outfile.jpg'

img = cv2.imread(filepath)

plt.figure(1)
plt.title('Original noised image')
plt.imshow(img)

# plt.figure(2)
# plt.title('2D Convolution ( Image Filtering )')
# kernal = np.ones((5, 5), np.float32) / 25
# filtered_image = cv2.filter2D(img, -1, kernal)
# plt.imshow(filtered_image)
#
# plt.figure(3)
# plt.title('Averaging')
# blur = cv2.blur(img, (5,5))
# plt.imshow(blur)

plt.figure(4)
plt.title('Gaussian blur')
blur = cv2.GaussianBlur(img, (5,5), 0)
# kernal = cv2.getGaussianKernel(10, 100)
# blur = cv2.filter2D(img, -1, kernal)
plt.imshow(blur)

# plt.figure(5)
# plt.title('Bilateral filtering')
# blur = cv2.bilateralFilter(img,9,75,75)
# plt.imshow(blur)

plt.show()