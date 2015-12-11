''' 
serial_fast.py 
Antuan Tran and Evan Yao, CS 205 Fall 2015 

This is a serial python-only implementation of our algorithm. We used 
it as a foundation for our parallel approach. This serial code is really slow! 

'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

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
            result.append((input[h,w], getWindow(input, (h,w), window_size).ravel()))
    return result


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

    for i in range(template.shape[0]):
        if mask[i]:
            count += 1
            total += (template[i] - image_chunk[i]) ** 2 * gaussian[i]

    return total / float(count)


def find_match(template, mask_chunk, windows, gaussian, delta = 0.3):
    '''
    Finds a match of the template in the mask
    '''
    # checking that the size of the template and mask are the same
    assert np.shape(template) == np.shape(mask_chunk)

    results = []

    # looping through all possible windows
    for (pixel, window) in windows:
        error = sum_square_error(window, template, mask_chunk, gaussian)
        results.append((error, pixel))

    errors, _ = zip(*results)
    return results


if __name__ == '__main__':
    def rgb2gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    try:
        image = rgb2gray(plt.imread('images/rings.jpg'))
    except:
        image = plt.imread('images/rings.jpg')

    window_size = 9

    # output image size
    output_h = 53
    output_w = 53

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
        #print mask
        new_mask, pixels_to_fill = fill_border(mask, top - i , left - i,  image.shape[0] + 2 * i)
        #print new_mask
        for x,y in pixels_to_fill:
            #print "Filling for (%d, %d) " % (x,y)
            mask_chunk = getWindow(mask, (x,y), window_size).ravel()
            pixelWindow = getWindow(output, (x,y), window_size).ravel()

            assert mask_chunk[mask_chunk.size / 2] == 0
            #print "Template"
            #print np.reshape(pixelWindow, (window_size,window_size))

            #print "Mask"
            #print np.reshape(mask_chunk, (window_size, window_size))
            possibleFill = find_match(pixelWindow, mask_chunk, image_windows, gaussian.ravel())

            possibleFill.sort(key=lambda x: x[0])
            output[x,y] = possibleFill[0][1]
            mask[x,y] = 1


        '''
        for x,y in pixels_to_fill:
            mask_chunk = getWindow(mask, (x,y), window_size)
            pixelWindow = getWindow(blank, (x,y), window_size)

            possibleFill = find_match(pixelWindow, mask_chunk, image, gaussian)

            candidateFill = []

            errors = []

            for error, pixel in possibleFill:
                errors.append(error)

            minError = min(errors) * 1.1

            for error, pixel in possibleFill:
                if error <= 1.1*minError:
                    candidateFill.append(pixel)

            #sys.exit(0)
            blank[x,y] = random.sample(candidateFill, 1)[0]
            mask[x,y] = 1
            progress += 1
            #raw_input("Press Enter to continue...")

        print "%d out of %d (%d %%)" % (progress, total , int(100*progress/total))

        '''
    plt.subplot(1,2,1)
    plt.imshow(output, cmap='Greys', interpolation='none')
    plt.title("Result")

    plt.subplot(1,2,2)
    plt.title("Template")
    plt.imshow(image, cmap='Greys', interpolation='none')
    plt.show()



