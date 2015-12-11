"""
functions.pyx
Antuan Tran and Evan Yao, CS 205 Fall 2015

Parallel Cython functions that help us do the major computations in
our algorithm. sum_square_error is for the slower of our 2 parallelization approaches
while find_matches_parallel is for the faster approach.
"""

from cython.parallel import prange

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sum_square_error(np.ndarray[np.double_t, ndim=1] template,
                       np.ndarray[np.double_t, ndim=1] image_chunk,
                       np.ndarray[np.double_t, ndim=1] mask,
                       np.ndarray[np.double_t, ndim=1] gaussian, int n):
    '''
    Returns the sum of square error of a particular template on
    top of an image chunk. Uses the provided gaussian for weighting.
    '''
    # checking sizes
    assert np.shape(template) == np.shape(image_chunk)
    assert np.shape(image_chunk) == np.shape(gaussian)
    assert np.shape(mask) == np.shape(image_chunk)

    cdef:
        double total = 0
        double count = 0
        int i

    for i in prange(template.shape[0],num_threads=n,nogil=True,schedule='static'):
            if mask[i]:
                count += 1
                total += (template[i] - image_chunk[i]) ** 2 * gaussian[i]

    return total / float(count)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef find_matches_parallel(np.ndarray[np.double_t, ndim=1] template,
                 np.ndarray[np.double_t, ndim=1] mask,
                 np.ndarray[np.double_t, ndim=1] image_windows,
                 np.ndarray[np.double_t, ndim=1] gaussian,
                np.ndarray[np.double_t, ndim=1] results, int n):

    cdef:
        int i, size_template, window_number, pixel_number

    # checking that the size of the template and mask are the same
    #assert np.shape(template) == np.shape(mask)
    size_template = template.shape[0]

    for i in prange(image_windows.shape[0], num_threads=n, nogil=True, schedule='guided'):

        window_number = i / size_template
        pixel_number = i % size_template
        if mask[pixel_number]:
            results[window_number] += (template[pixel_number] - image_windows[i]) ** 2 * gaussian[pixel_number]

    return results[:]
