#!/usr/bin/env python

from __future__ import print_function
import time
import numpy as np
import theano
import theano.tensor as T


X = T.matrix()
Y = T.matrix()

Z = X.dot(Y)
f = theano.function([X, Y], Z)

x = np.random.rand(10000, 10000)
y = np.random.rand(10000, 10000)

start = time.time()
z = f(x, y)
end = time.time()

print('time spent: {0} second'.format(end - start))
