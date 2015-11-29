#
# @author  Daniel Maturana
# @year    2015
#
# @attention Copyright (c) 2015
# @attention Carnegie Mellon University
# @attention All rights reserved.
#
# @=

import numpy as np

import lasagne
from lasagne.utils import floatX

class Prelu(lasagne.init.Initializer):
    def __init__(self):
        pass

    def sample(self, shape):
        # eg k^2 for conv2d
        receptive_field_size = np.prod(shape[2:])
        c = shape[1] # input channels
        nl = c*receptive_field_size
        std = np.sqrt(2.0/(nl))
        return floatX(np.random.normal(0, std, size=shape))
