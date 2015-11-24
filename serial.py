import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import sys
import random


def find_match(template, mask, image):
    '''
    Finds a match of the template in the mask
    '''
    # checking that the size of the template and mask are the same
    assert np.shape(template) == np.shape(mask)

    # size variables
    (h_t, w_t) = np.shape(template)
    (h_i, w_i) = np.shape(image)

    results = []
    # looping through all possible windows
    for h in range(h_i - h_t):
        for w in range(w_i - w_t):
            results.append(sum_square_error(template, image[], mask[]), image[h,w])

    return results


def sum_square_error(template, image_chunk, mask):
    global gaussian

    '''
    Returns the sum of square error of a particular template on
    top of an image chunk. Uses the provided gaussian for weighting.
    '''
    # checking sizes
    assert np.shape(template) == np.shape(image_chunk)
    assert np.shape(image_chunk) == np.shape(gaussian)
    assert np.shape(mask) == np.shape(image_chunk)

    total = 0

    for i in range(np.shape(template)[0]):
        for j in range(np.shape(template)[1]):
            if mask[i,j]:
                total += (template[i,j] - image_chunk[i,j]) ** 2) * gaussian[i,j]

    return total


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def dilate(image):
    '''
    TODO
    '''
    dilatedList = []
    dilated = np.copy(image)
    (h,w) = np.shape(image)
    for i in range(h):
        for j in range(w):
            if image[i,j] == 1:
                if i > 0:
                    dilated[i - 1, j] = 1
                if i < h - 1:
                    dilated[i + 1, j] = 1
                if j > 0:
                    dilated[i, j - 1] = 1
                if j < w - 1:
                    dilated[i, j + 1] = 1
    for i in range(h):
        for j in range(w):
            if image[i,j] == 0 and dilated[i,j] == 1:
                dilatedList.append((i,j))
    return dilatedList


if __name__ == '__main__':
    '''
    Playing, Testing Code
    '''

    # test_image = np.zeros((11, 11))
    # test_template = np.zeros((5,5))

    # test_image[0:11:2,0:11:2] = 1
    # test_image[1:11:2,1:11:2] = 1

    # test_template[0:5:2, :] = 1
    # test_template[:, 1:11:2] = 1

    # plt.imshow(test_template, cmap='Greys', interpolation='none')
    # plt.show()

    # plt.imshow(test_image, cmap = 'Greys', interpolation='none')
    # plt.show()

    # if len(sys.argv) != 1:
    #     sys.exit(0)
    '''
    Pre-processing: loading image, setting up template
    '''

    image = plt.imread('images/sand_template.gif')

    (h, w) = (300, 300)

    window_size = 5

    (imheight, imwidth, color) = np.shape(image)

    blank = np.zeros((h, w, color), dtype=np.uint8)

    top = (h - imheight) / 2
    left = (w - imwidth) / 2

    blank[top:top + imheight, left:left + imwidth, :] = image

    plt.imshow(blank)
    plt.show()

    # pixels that have been filled in
    mask = np.zeros((h,w), dtype=float)
    mask[top:top + imheight, left:left + imwidth] = 1

    pixels_to_fill = dilate(mask)

    for x,y in pixels_to_fill:
        ptop = x - (window_size / 2)
        pleft = y - (window_size / 2)
        pixelWindow = blank[ptop:ptop + window_size, pleft:pleft + window_size]
        pixelMask = mask[ptop:ptop + window_size, pleft:pleft + window_size]
        possibleFill = find_match(pixelWindow, pixelMask, image)
        candidateFill = []
        errors = []
        for error, pixel in possibleFill:
            errors.append(error)
        minError = min(errors) * 1.1
        for error, pixel in possibleFill:
            if error < minError:
                candidateFill.append(pixel)
        blank[x,y] = random.sample(candidateFill)


    plt.imshow(mask)
    plt.show()


# test for dilate
#mask[top + 25:top + imheight - 25, left + 25:left + imwidth - 25] = 0
    #test = np.copy(mask)
    #plt.imshow(test, cmap = 'Greys', interpolation = 'none')
    #print dilate(test)
    

