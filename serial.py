import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    '''
    Pre-processing: loading image, setting up template
    '''

    image = plt.imread('images/sand_template.gif')

    (h, w) = (300, 300)

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

    # dilated pixes

    plt.imshow(mask)
    plt.show()


def dilate(image):
    '''
    TODO
    '''

def sum_square_diff(template, mask, image):
    '''
    TODO
    '''


