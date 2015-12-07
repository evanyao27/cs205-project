import sys
sys.path.append('util')

import set_compiler
set_compiler.install()

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

from timer import Timer
from functions import find_matches
from serial import gkern

t = 80
s = t*t

test_template = np.ones(s)
test_windows = np.array(
    [range(s) for _ in range(s)]
, dtype=np.double).ravel()

test_mask = np.zeros(s)
test_gaussian = np.ones(s)
results = np.zeros(s)

test_windows = test_windows.ravel()

total = 0
for i in [1,2,4,8,16, 32]:
    print "Num Threads: %d " % i
    total = 0
    for _ in range(10):
        results = np.zeros(s)
        with Timer() as t:
            find_matches(test_template, test_mask, test_windows, test_gaussian, results, i)
        total += t.interval
    print total
#print results
#print np.sum(np.power (np.subtract(test_template, test_windows[0]), 2))
