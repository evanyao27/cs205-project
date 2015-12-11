''' 
helpers.py 
Antuan Tran and Evan Yao, CS 205 Fall 2015


Python helper functions that are used in implementing the algorithm. 
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import sys
import scipy.stats as st
import random
import time

def rgb2gray(rgb):
    '''
    Tries to convert an rgb image to greyscale
    '''
    try:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    except:
        return rgb


def gkern(kernlen=21, nsig=3):
    '''
    Returns a 2-D square Gaussian Kerenel with a given standard deviation
    '''

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def getWindow(image, center, winSize):
    '''
    Extracts a particular window centered around a particular pixel, but
    pads with 0's if the window extends beyond the border of the image.
    '''

    (h, w) = np.shape(image)
    x,y = center
    side = (winSize / 2)
    window = np.zeros((winSize,winSize), dtype=float)

    # set boundaries of image window
    top = x - side
    bottom = x + side + 1
    left = y - side
    right = y + side + 1

    # boundaries of window
    wTop = 0
    wBottom = winSize
    wLeft = 0
    wRight = winSize

    # keep image boundary within image
    # also update into where we copy the image
    if top < 0:
        wTop = 0 - top
        top = 0
    if bottom > h - 1:
        wBottom = wBottom - (bottom - h)
        bottom = h
    if left < 0:
        wLeft = 0 - left
        left = 0
    if right > w - 1:
        wRight = wRight - (right - w)
        right = w

    window[wTop:wBottom, wLeft:wRight] = image[top:bottom, left:right]

    return window


def fill_border(image, top, left, width):
    '''
    Takes a binary image and fills in a frame whose left and top indices
    are given, and then a particular width for the square hole is also given.

    We use this function to dilate the mask of already filled pixels in order
    to find the next set of (x,y) pixel locations to fill.
    '''

    bottom = top + width - 1
    right = left + width - 1

    indices = []

    for i in range(width):
        if top >= 0:
            # image[top, left + i] = 1
            # image[bottom, left + i] = 1

            if i != 0 and i != width - 1:
                indices.append((top, left + i))
                indices.append((bottom, left + i))

        if left >= 0:
            # image[top + i, left] = 1
            # image[top + i, right] = 1

            indices.append((top + i, left))
            indices.append((top + i, right))

    return image, indices

def get_simple_test():
    test_image = np.zeros((9,9))

    for (x,y) in [
        (1,0),
        (1,1),
        (1,2),
        (1,4),
        (1,5),
        (1,6),
        (2,0),
        (2,2),
        (2,4),
        (2,6),
    ]:
        test_image[x+1,y+1] = 255
        test_image[x+5,y+1] = 255

    return test_image