import sys
import os.path
sys.path.append('util')

import set_compiler
set_compiler.install()

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

from timer import Timer
from functions import sum_square_error
import scipy.stats as st

s = 10000

template = np.ones((s,s), dtype=np.double)
image = np.zeros((s,s), dtype=np.double)
mask = np.ones((s,s), dtype=np.double)

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

gaussian = gkern(s, 3)
gaussian = np.double(gaussian)

with Timer() as t:
    sum_square_error(template.ravel(), image.ravel(), mask.ravel(), gaussian.ravel())
print t.interval

from serial import sum_square_error
with Timer() as t:
    sum_square_error(template, image, mask, gaussian)
print t.interval

