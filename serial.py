import numpy as np
import matplotlib.pyplot as plt

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

    


    plt.imshow(mask)
    plt.show()



def sum_square_diff(template, mask, image):
    '''
    TODO
    '''


# test for dilate
#mask[top + 25:top + imheight - 25, left + 25:left + imwidth - 25] = 0
    #test = np.copy(mask)
    #plt.imshow(test, cmap = 'Greys', interpolation = 'none')
    #print dilate(test)
    