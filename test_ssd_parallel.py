import sys
sys.path.append('util')

import set_compiler
set_compiler.install()

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

from timer import Timer
from functions import sum_square_error
from serial import gkern

s = 10000

template = np.ones((s,s), dtype=np.double)
image = np.zeros((s,s), dtype=np.double)
mask = np.ones((s,s), dtype=np.double)

gaussian = gkern(s, 3)
gaussian = np.double(gaussian)


for i in range(1, 12):
    print "Num Threads: %d " % i
    total = 0
    for _ in range(1):
        with Timer() as t:
            sum_square_error(template.ravel(), image.ravel(), mask.ravel(), gaussian.ravel(), i)
        total += t.interval
    print total


