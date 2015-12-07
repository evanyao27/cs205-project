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

results = []
for i in range(1, 8):
    print "Num Threads: %d " % i
    total = 0
    for _ in range(100000/s):
        with Timer() as t:
            sum_square_error(template.ravel(), image.ravel(), mask.ravel(), gaussian.ravel(), i)
        total += t.interval
    results.append(total)

import matplotlib.pyplot as plt
plt.plot(range(1,8), results)

plt.title("Subprocess: Difference of Squares (%d x %d window)" % (s, s))
plt.ylabel("Time for %d calculations" % (10000/s))
plt.xlabel("Number of Threads")
plt.show()