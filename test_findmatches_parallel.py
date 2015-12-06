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

t = 5
s = t*t


test_template = np.ones(s)
test_windows = np.array(
    [range(s) for _ in range(s)]
, dtype=np.double)

test_mask = np.ones(s)
test_gaussian = np.ones(s)
results = np.zeros(s)

total = 0
for i in range(1, 12):
    print "Num Threads: %d " % i
    total = 0
    for _ in range(100):
        results = np.zeros(s)
        with Timer() as t:
            find_matches(test_template, test_mask, test_windows, test_gaussian, results, i)
        total += t.interval
    print total