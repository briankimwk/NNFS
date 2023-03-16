# This aims to output a number between 0 and 1 like a probability
import math
import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]


exp_values = np.exp(layer_outputs)
norm_base = np.sum(exp_values, axis=1, keepdims=True)

norm_values = exp_values / norm_base
print(norm_values)

