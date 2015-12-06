import sys
import os.path
sys.path.append('util')

import set_compiler
set_compiler.install()

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

from timer import Timer
from functions import sum_square_error, find_matches

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import sys
import scipy.stats as st
import random
import time

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def getWindow(image, center, winSize):
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
        wBottom = wBottom - (bottom - h + 1)
        bottom = h - 1
    if left < 0:
        wLeft = 0 - left
        left = 0
    if right > w - 1:
        wRight = wRight - (right - w + 1)
        right = w - 1

    window[wTop:wBottom, wLeft:wRight] = image[top:bottom, left:right]

    return window

def get_valid_windows(input, window_size):
    '''
    :param input: an NxM numpy array representing a static image
    :param window_size: size of the window we're grabbing from the input
    :return: list of [(pixel, [neighborhood]), (pixel, [neighborhood])...]
    '''

    result = []
    for h in range(input.shape[0]):
        for w in range(input.shape[1]):
            result.append(getWindow(input, (h,w), window_size).ravel())

    return np.array(result)


def fill_border(image, top, left, width):
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

def find_match(template, mask_chunk, windows, gaussian, delta = 0.3):
    '''
    Finds a match of the template in the mask
    '''
    # checking that the size of the template and mask are the same
    assert np.shape(template) == np.shape(mask_chunk)

    # size variables
    window_size = template.shape[0]
    (h_i, w_i) = np.shape(image)

    results = []

    # looping through all possible windows
    for window in windows:
        error = sum_square_error(window, template, mask_chunk, gaussian)
        results.append(error)

    return results

def sum_square_error(template, image_chunk, mask, gaussian):
    #global gaussian
    '''
    Returns the sum of square error of a particular template on
    top of an image chunk. Uses the provided gaussian for weighting.
    '''
    # checking sizes
    assert np.shape(template) == np.shape(image_chunk)
    assert np.shape(image_chunk) == np.shape(gaussian)
    assert np.shape(mask) == np.shape(image_chunk)

    total = 0

    for i in range(template.shape[0]):
        if mask[i]:
            total += (template[i] - image_chunk[i]) ** 2 * gaussian[i]

    return total

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

try:
    image = rgb2gray(plt.imread('images/rings.jpg'))
except:
    image = plt.imread('images/rings.jpg')

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

#image = test_image
window_size = 9

# output image size
output_h = 103
output_w = 103

# if we're trying to synthesize something smaller than our template
if output_h <= image.shape[0] or output_w < image.shape[1]:
    raise Exception("Dimensions of synthesized image must be greater than that of input!")

# our output image
output = np.zeros((output_h, output_w), dtype=float)

# getting the corner coordinates of the image patch inside the blank (for now) output
top = (output_h - image.shape[0]) / 2
left = (output_w - image.shape[1]) / 2

# plant the image inside the blank
output[top:top + image.shape[0], left:left + image.shape[1]] = image

# windows to try and match
image_windows = get_valid_windows(image, window_size)

# binary matrix of pixels that have already been filled in
mask = np.zeros((output_h,output_w), dtype=float)
mask[top:top + image.shape[0], left:left + image.shape[1]] = 1

gaussian = gkern(window_size, 3)

progress = 0
total = np.size(output) - np.size(image)

for i in range(1, (output_h - image.shape[0])/2 + 1):
    print image_windows.size
    print i, " out of ", (output_h - image.shape[0])/2 + 1
    new_mask, pixels_to_fill = fill_border(mask, top - i , left - i,  image.shape[0] + 2 * i)
    for x,y in pixels_to_fill:
        mask_chunk = getWindow(mask, (x,y), window_size).ravel()
        pixelWindow = getWindow(output, (x,y), window_size).ravel()

        assert mask_chunk[mask_chunk.size / 2] == 0

        possibleFill = np.zeros(image_windows.shape[0])

        find_matches(pixelWindow, mask_chunk, image_windows, gaussian.ravel(), possibleFill, 1)

        possibleFill2 = find_match(pixelWindow, mask_chunk, image_windows, gaussian.ravel())
        print possibleFill
        print possibleFill2
        print possibleFill - possibleFill2
        assert (possibleFill == possibleFill2).all()
        small_error_index = list(possibleFill2).index(min(possibleFill2))
        output[x,y] = image_windows[small_error_index][image_windows.shape[1]/2]

        mask[x,y] = 1
#
plt.subplot(1,2,1)
plt.imshow(output, cmap='Greys', interpolation='none')
plt.title("Result")

plt.subplot(1,2,2)
plt.title("Template")
plt.imshow(image, cmap='Greys', interpolation='none')
plt.show()






