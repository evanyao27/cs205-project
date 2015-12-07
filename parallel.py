import sys
sys.path.append('util')

import set_compiler
set_compiler.install()

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

from helpers import *
from timer import Timer
from functions import sum_square_error, find_matches

import numpy as np
import matplotlib.pyplot as plt
import time


def get_valid_windows(input, window_size):
    '''
    :param input: an NxM numpy array representing a static image
    :param window_size: size of the window we're grabbing from the input
    :return: list of [(pixel, [neighborhood]), (pixel, [neighborhood])...]
    '''

    result = []
    #print window_size
    for h in range(input.shape[0]):
        for w in range(input.shape[1]):
            result.append(getWindow(input, (h,w), window_size).ravel())

    return np.array(result)



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

try:
    image = rgb2gray(plt.imread('images/text_template.gif')).astype(float)
except:
    image = plt.imread('images/text_template.gif').astype(float)


def synthesize(input_image, output_size, window_size, epsilon = 0.1,
               delta = 0.3, verbose = False, num_threads = 4):
    """
    Main function which returns a new image of shape 'output_size' synthesized
     with the given additional parameters

    :param input_image: np 2-D array representing the square static image
    :param output_size: size of the square output image
    :param window_size: size of the window used for finding matches
    :param epsilon: any pixel whose window's error is within (1+epsilon)
    :param delta: if we find a pixel whose window error less than delta, we will case
     finding new candidates.
    :param verbose: whether we should print progress
    :return: a newly synthesized matrix
    """

    isColor = len(input_image.shape) > 2

    if isColor:
        gray_image = rgb2gray(input_image)
        color_image = input_image[:,:,0:3]
    else:
        gray_image = input_image

    # if we're trying to synthesize something smaller than our template
    if output_size < gray_image.shape[0] or output_size < gray_image.shape[1]:
        raise Exception("Dimensions of synthesized image must be greater than that of input!")

    # getting the corner coordinates of the image patch inside the blank (for now) output
    top = (output_size - gray_image.shape[0]) / 2
    left = (output_size - gray_image.shape[1]) / 2

    # our output image
    output = np.zeros((output_size, output_size), dtype=float)
    if isColor:
        color_output = np.zeros((output_size, output_size, 3))
        color_output[top:top + color_image.shape[0], left:left + color_image.shape[1], 0:3] = color_image

    # plant the image inside the blank
    output[top:top + gray_image.shape[0], left:left + gray_image.shape[1]] = gray_image

    # windows to try and match
    image_windows = get_valid_windows(gray_image, window_size)

    # binary matrix of pixels that have already been filled in
    mask = np.zeros((output_size,output_size), dtype=float)
    mask[top:top + gray_image.shape[0], left:left + gray_image.shape[1]] = 1

    gaussian = gkern(window_size, 3)

    #print image_windows.shape
    #raise Exception("stop")
    for i in range(1, (output_size - gray_image.shape[0])/2 + 1):
        if verbose:
            print i, " out of ", (output_size - gray_image.shape[0])/2 + 1

        new_mask, pixels_to_fill = fill_border(mask, top - i, left - i,  gray_image.shape[0] + 2 * i)

        for x,y in pixels_to_fill:
            mask_chunk = getWindow(mask, (x,y), window_size).ravel()
            pixelWindow = getWindow(output, (x,y), window_size).ravel()

            possibleFill = np.zeros(image_windows.shape[0])

            find_matches(pixelWindow, mask_chunk, image_windows.ravel(), gaussian.ravel(), possibleFill, num_threads)

            index = np.random.choice(np.flatnonzero(possibleFill <= 1.1 * np.min(possibleFill)), 1)[0]

            output[x,y] = image_windows[index][image_windows.shape[1]/2]

            if isColor:
                color_output[x,y,0] = color_image[index / color_image.shape[0], index % color_image.shape[0], 0]
                color_output[x,y,1] = color_image[index / color_image.shape[0], index % color_image.shape[0], 1]
                color_output[x,y,2] = color_image[index / color_image.shape[0], index % color_image.shape[0], 2]

            mask[x,y] = 1

    if isColor:
        return color_output
    else:
        return output

def time_this(threads):
    color_image = plt.imread('images/scale_template.gif')
    end = 49

    try:
        color_image = color_image[:end,:end,:]
    except:
        color_image = color_image[:end,:end]

    with Timer() as t:
        new_image = synthesize(color_image, 89, 17, verbose = False, num_threads=threads)
    print t.interval
    return t.interval

    # plt.subplot(1,2,1, aspect='equal')
    # plt.imshow(new_image.astype(np.uint8), cmap='Greys', interpolation='none')
    # plt.title("Result")
    #
    # plt.subplot(1,2,2, aspect='equal')
    # plt.title("Template")
    # plt.imshow(color_image, interpolation='none', cmap='Greys')
    # plt.show()

times = [time_this(i) for i in range(1, 9)]
plt.plot(range(1,9), [34.645647049,
35.4726171494,
32.721255064,
33.8481218815,
34.9053170681,
33.9290750027,
34.4410350323,
33.2712552292,
])
plt.title("Running Time for Overall Algorithm")
plt.xlabel("Number of Threads")
plt.ylabel("Time")
plt.show()

