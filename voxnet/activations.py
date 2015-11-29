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
softplus = T.nnet.softplus
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
