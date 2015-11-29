#
# @author  Daniel Maturana
# @year    2015
#
# @attention Copyright (c) 2015
# @attention Carnegie Mellon University
# @attention All rights reserved.
#
# @=

"""
Activation functions

some are redundant as they have been implemented in lasagne.
"""

# TODO maybe steal from pylearn2

import theano.tensor as T

__all__ = [\
        'tanh',
        'sigmoid',
        'softplus',
        'softmax',
        'relu'
        ]

tanh = T.tanh
sigmoid = T.nnet.sigmoid
# softplus is log(1+exp(x))
softplus = T.nnet.softplus
# rowwise softmax
softmax = T.nnet.softmax

def relu(x):
    """Rectified linear units (relu)"""
    return T.maximum(0,x)

def make_leaky_relu(leakiness=0.01):
    def leaky_relu(x):
        return T.maximum(leakiness*x,x)
    return leaky_relu

def leaky_relu_001(x):
    """ hard-coded leaky relu """
    return T.maximum(0.01*x,x)

def leaky_relu_01(x):
    """ hard-coded leaky relu """
    return T.maximum(0.1*x,x)

def leaky_relu_03(x):
    """ hard-coded leaky relu """
    return T.maximum(0.3*x,x)

# rectify from lasagne
# The following is faster than lambda x: T.maximum(0, x)
# Thanks to @SnippyHolloW for pointing this out.
# See: https://github.com/SnippyHolloW/abnet/blob/807aeb98e767eb4e295c6d7d60ff5c9006955e0d/layers.py#L15
#relu = lambda x: (x + abs(x)) / 2.0
# FROM DMS: relu seems slightly faster?
