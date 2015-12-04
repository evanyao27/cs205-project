from cython.parallel import parallel, prange
from openmp cimport omp_lock_t, \
    omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock, omp_get_thread_num
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np


cpdef find_match(np.int8[:,:] template, np.int8[:,:] mask, np.int8[:,:] image):
	cdef int h_t, w_t, h_i, w_i, h, w
		np.int8[:,:] results
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
            results.append((sum_square_error(template, image[h:h+h_t, w:w+w_t],
                                            mask), image[h,w]))
    return results

