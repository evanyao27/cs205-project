from cython.parallel import parallel, prange
from openmp cimport omp_lock_t, \
    omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock, omp_get_thread_num
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

cpdef sum_square_error(np.double_t[:] template,
                       np.double_t[:] image_chunk,
                       np.double_t[:] mask,
                       np.double_t[:] gaussian):
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

    for i in prange(template.shape[0],num_threads=1,nogil=True,schedule='static'):
            if mask[i]:
                count += 1
                total += (template[i] - image_chunk[i]) ** 2 * gaussian[i]

    return total / float(count)


cpdef find_match(np.double_t[:] template,
                 np.double_t[:] mask,
                 np.double_t[:,:] image_windows,
                 np.double_t[:] gaussian):

    cdef:
        int pixel, i, num_windows, size_template, window_number, pixel_number
        np.ndarray results = np.zeros([num_windows*size_template], dtype = np.double_t)


    # checking that the size of the template and mask are the same
    assert np.shape(template) == np.shape(mask)
    #results = <np.double_t *>malloc(num_windows*size_template*sizeof(np.double_t))
    
    
    num_windows = image_windows.shape[0]
    size_template = template.shape[0]

    # looping through all possible windows
    for i in prange(num_windows * size_template, num_threads=1, nogil=True, schedule='static'):
        window_number = i / num_windows
        pixel_number = i % size_template

        results[window_number] += (template[pixel_number] - image_windows[window_number][pixel_number]) ** 2 * gaussian[pixel_number]

    return results