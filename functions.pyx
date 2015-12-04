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

    for i in prange(template.shape[0],num_threads=4,nogil=True,schedule='static'):
            if mask[i]:
                count += 1
                total += (template[i] - image_chunk[i]) ** 2 * gaussian[i]

    return total / float(count)

'''
cpdef find_match(np.int8[:,:] template, np.int8[:,:] mask, np.int8[:,:] image):
	cdef int h_t, w_t, h_i, w_i, h, w
		np.int8[:,:] results

    Finds a match of the template in the mask

    # checking that the size of the template and mask are the same
    assert np.shape(template) == np.shape(mask)

    # size variables
    (h_t, w_t) = np.shape(template)
    (h_i, w_i) = np.shape(image)

    results = []

    # looping through all possible windows
    for h in range(h_i - h_t):
        for w in range(w_i - w_t):
            results.append((sum_square_error(template, image[h:h+h_t, w:w+w_t],
                                            mask), image[h,w]))
    return results
'''