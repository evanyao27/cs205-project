import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import sys
import scipy.stats as st
import random
import time


def find_match(template, mask_chunk, image, delta = 0.3):
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
    for h in range(h_i):
        for w in range(w_i):
            image_chunk = getWindow(image, (h,w), window_size)
            error = sum_square_error(template, image_chunk, mask_chunk)
            pix = image[h,w]
            results.append((error, pix))
            if error < delta:
                break
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
    count = 0

    for i in range(np.shape(template)[0]):
        for j in range(np.shape(template)[1]):
            if mask[i,j]:
                count += 1
                total += (template[i,j] - image_chunk[i,j]) ** 2 * gaussian[i,j]

    return total / float(count)


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
    Dilates a binary image
    '''
    dilatedList = []
    dilated = np.copy(image)
    (h,w) = np.shape(image)

    for i in range(h):
        for j in range(w):
            if image[i,j] == 1:
                # up/down/left/right
                if i > 0:
                    dilated[i - 1, j] = 1
                if i < h - 1:
                    dilated[i + 1, j] = 1
                if j > 0:
                    dilated[i, j - 1] = 1
                if j < w - 1:
                    dilated[i, j + 1] = 1

                # corners
                if i > 0 and j > 0:
                    dilated[i - 1, j - 1] = 1
                if i > 0 and j < w - 1:
                    dilated[i - 1, j + 1] = 1
                if i < h - 1 and j > 0:
                    dilated[i + 1, j - 1] = 1
                if i < h - 1 and j < w - 1:
                    dilated[i + 1, j + 1] = 1

    for i in range(h):
        for j in range(w):
            if image[i,j] == 0 and dilated[i,j] == 1:
                dilatedList.append((i,j))

    return dilatedList

# assume window size is odd
def getWindow(image, center, winSize):

    (h, w) = np.shape(image)
    x,y = center
    side = (winSize / 2)
    window = np.zeros((winSize,winSize), dtype=np.uint8)

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

if __name__ == '__main__':

    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    try:
        image = rgb2gray(plt.imread('images/rings.jpg'))
    except:
        image = plt.imread('images/rings.jpg')
    image = image

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

    image = test_image


    (h, w) = (41, 41)

    window_size = 15

    (imheight, imwidth) = np.shape(image)

    blank = np.zeros((h, w), dtype=np.uint8)

    top = (h - imheight) / 2
    left = (w - imwidth) / 2

    blank[top:top + imheight, left:left + imwidth] = image

    # pixels that have been filled in
    mask = np.zeros((h,w), dtype=float)
    mask[top:top + imheight, left:left + imwidth] = 1

    gaussian = gkern(window_size, 3)

    progress = 0
    total = np.size(blank) - np.size(image)

    while True:
        pixels_to_fill = dilate(mask)
        if len(pixels_to_fill) == 0:
            break
        for x,y in pixels_to_fill:
            mask_chunk = getWindow(mask, (x,y), window_size)
            pixelWindow = getWindow(blank, (x,y), window_size)

            possibleFill = find_match(pixelWindow, mask_chunk, image)

            candidateFill = []

            errors = []

            for error, pixel in possibleFill:
                errors.append(error)

            minError = min(errors) * 1.1
            print minError
            for error, pixel in possibleFill:
                if error <= 1.1*minError:
                    candidateFill.append(pixel)

            #sys.exit(0)
            blank[x,y] = random.sample(candidateFill, 1)[0]
            mask[x,y] = 1
            progress += 1
            #raw_input("Press Enter to continue...")

        print "%d out of %d (%d %%)" % (progress, total , int(100*progress/total))

    plt.subplot(1,2,1)
    plt.imshow(blank, cmap='Greys', interpolation='none')

    plt.subplot(1,2,2)
    plt.imshow(image, cmap='Greys', interpolation='none')
    plt.show()



