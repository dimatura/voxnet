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
Layers for spatially 3d/ volumetric data
"""

import numpy as np

import lasagne
from lasagne.layers import Layer

from .max_pool_3d import max_pool_3d

import theano
import theano.tensor as T
from theano.tensor.nnet import conv3d2d
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.blas import GpuCorr3dMM

floatX = theano.config.floatX

__all__ = [
        'Conv3dLayer',
        'Conv3dMMLayer',
        'RotPool3dLayer',
        'SplitLayer',
        ]

class Conv3dLayer(Layer):
    """
    TODO lasagne conventions for padding etc
    """
    def __init__(self, input_layer, num_filters, filter_size,
            border_mode=None,
            W=lasagne.init.Normal(std=0.01),
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad=None,
            **kwargs):
        """
        input_shape: (frames, height, width)
        W_shape: (out, in, kern_frames, kern_height, kern_width)
        """
        super(Conv3dLayer, self).__init__(input_layer, **kwargs)

        # TODO note that lasagne allows 'untied' biases, the same shape
        # as the input filters.

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = (1,1,1)
        self.flip_filters = False
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # TODO lasagne calc size

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and 'pad'. To avoid ambiguity, please specify only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = (0, 0, 0)
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = (0, 0, 0)
            elif border_mode == 'full':
                self.pad = (self.filter_size[0] - 1, self.filter_size[1] -1, self.filter_size[2] - 1)
            elif border_mode == 'same':
                # only works for odd filter size, but the even filter size case is probably not worth supporting.
                self.pad = ((self.filter_size[0]-1) // 2,
                            (self.filter_size[1]-1) // 2,
                            (self.filter_size[2]-1) // 2)
            else:
                raise RuntimeError("Unsupported border_mode for Conv3dLayer: %s" % border_mode)
        else:
            self.pad = pad

        self.W = self.add_param(W, self.get_W_shape(), name='W')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_filters,), name='b', regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_layer.get_output_shape()[1]
        return (self.num_filters, num_input_channels, self.filter_size[0], self.filter_size[1], self.filter_size[2])

    def get_output_shape_for(self, input_shape):
        """ input is bct01
        """
        batch_size = input_shape[0]
        volume_shape = np.asarray(input_shape[-3:]).astype(np.float32)
        filter_size = np.asarray(self.filter_size).astype(np.float32)
        pad = np.asarray(self.pad).astype(np.float32)
        out_dim = ( (volume_shape + 2*pad - filter_size) + 1 ).astype(np.int32)
        return (batch_size, self.num_filters, out_dim[0], out_dim[1], out_dim[2])

    def get_output_for(self, input, *args, **kwargs):
        """ input is bct01
        based on
        https://github.com/lpigou/Theano-3D-ConvNet/blob/master/convnet3d/convnet3d.py
        released as public domain.
        """
        input_shape = self.input_layer.get_output_shape()
        t, h, w = input_shape[2], input_shape[3], input_shape[4]
        input_c = input_shape[1]
        batch_size = input_shape[0]
        filter_t, filter_h, filter_w = self.filter_size
        input_btc01 = input.dimshuffle([0,2,1,3,4]) # bct01 -> btc01
        out_btc01 = conv3d2d.conv3d(signals=btc01, filters=self.W,
                signals_shape=(batch_size, t, input_c, h, w),
                filters_shape=(self.num_filters, filter_t, input_c, filter_h, filter_w),
                border_mode='valid')
        out_bct01 = out_btc01.dimshuffle([0,2,1,3,4]) # btc01 -> bct01
        if self.b is not None:
            out_bct01 = out_bct01 + self.b.dimshuffle('x',0,'x','x','x')
        return self.nonlinearity(out_bct01)

class Conv3dMMLayer(Layer):
    def __init__(self, input_layer, num_filters, filter_size,
            strides=(1,1,1),
            border_mode=None,
            W=lasagne.init.Normal(std=0.001), # usually 0.01
            b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad=None,
            flip_filters=True,
            **kwargs):
        """
        input_shape: (frames, height, width)
        W_shape: (out, in, kern_frames, kern_height, kern_width)
        """
        super(Conv3dMMLayer, self).__init__(input_layer, **kwargs)

        # TODO note that lasagne allows 'untied' biases, the same shape
        # as the input filters.
        self.num_filters = num_filters
        self.filter_size = filter_size
        if strides is None:
            self.strides = (1,1,1)
        else:
            self.strides = tuple(strides)
        self.flip_filters = flip_filters
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if border_mode is not None and pad is not None:
            raise RuntimeError("You cannot specify both 'border_mode' and 'pad'. To avoid ambiguity, please specify only one of them.")
        elif border_mode is None and pad is None:
            # no option specified, default to valid mode
            self.pad = (0, 0, 0)
        elif border_mode is not None:
            if border_mode == 'valid':
                self.pad = (0, 0, 0)
            elif border_mode == 'full':
                self.pad = (self.filter_size[0] - 1, self.filter_size[1] -1, self.filter_size[2] - 1)
            elif border_mode == 'same':
                # only works for odd filter size, but the even filter size case is probably not worth supporting.
                self.pad = ((self.filter_size[0] - 1) // 2,
                            (self.filter_size[1] - 1) // 2,
                            (self.filter_size[2] - 1) // 2)
            else:
                raise RuntimeError("Unsupported border_mode for Conv3dLayer: %s" % border_mode)
        else:
            self.pad = tuple(pad)

        self.W = self.add_param(W, self.get_W_shape(), name='W')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_filters,), name='b', regularizable=False)
        self.corr_mm_op = GpuCorr3dMM(subsample=self.strides, pad=self.pad)

    def get_W_shape(self):
        # out in t01
        #num_input_channels = self.input_layer.get_output_shape()[1]
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0], self.filter_size[1], self.filter_size[2])

    def get_output_shape_for(self, input_shape):
        """ input is bct01
        """
        batch_size = input_shape[0]
        volume_shape = np.asarray(input_shape[-3:]).astype(np.float32)
        filter_size = np.asarray(self.filter_size).astype(np.float32)
        pad = np.asarray(self.pad).astype(np.float32)
        strides = np.asarray(self.strides).astype(np.float32)
        # TODO check this is right. also it depends on border_mode
        # this assumes strides = 1
        #out_dim = (video_shape-(2*np.floor(kernel_shape/2.))).astype(np.int32)
        #out_dim = ( (volume_shape-filter_size) + 1).astype(np.int32)
        out_dim = ( (volume_shape + 2*pad - filter_size) // strides + 1 ).astype(np.int32)
        return (batch_size, self.num_filters, out_dim[0], out_dim[1], out_dim[2])

    def get_output_for(self, input, *args, **kwargs):
        # TODO figure out shapes
        # TODO filter flip
        t, h, w = input.shape[2], input.shape[3], input.shape[4]
        filter_t, filter_h, filter_w = self.filter_size
        #input_c = input.shape[1]

        filters = self.W
        if self.flip_filters:
            filters = filters[:,:,::-1,::-1,::-1]
        contiguous_filters = gpu_contiguous(filters)
        contiguous_input = gpu_contiguous(input)

        conved = self.corr_mm_op(contiguous_input, contiguous_filters)

        if self.b is None:
            activation = conved
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x', 'x')

        return self.nonlinearity(activation)


class MaxPool3dLayer(Layer):
    def __init__(self, input_layer, pool_shape, **kwargs):
        super(MaxPool3dLayer, self).__init__(input_layer, **kwargs)
        self.pool_shape = pool_shape

    def get_output_shape_for(self, input_shape):
        # if ignore_border=False (default), out is ceil; else it's floor
        return (input_shape[0],
                input_shape[1],
                int(np.ceil(float(input_shape[2])/(self.pool_shape[0]))),
                int(np.ceil(float(input_shape[3])/(self.pool_shape[1]))),
                int(np.ceil(float(input_shape[4])/(self.pool_shape[2]))))

    def get_output_for(self, input, *args, **kwargs):
        out = max_pool_3d(input, self.pool_shape)
        self.output = out
        return self.output

class RotPool3dLayer(Layer):
    """
    Pool over num_degs rotations.
    """
    def __init__(self, input_layer, num_degs, method='max', **kwargs):
        super(RotPool3dLayer, self).__init__(input_layer, **kwargs)
        self.method = method
        self.num_degs = num_degs

    # rely on defaults for get_params
    def get_output_shape_for(self, input_shape):
        assert( len(input_shape)==2 )
        if any(s is None for s in input_shape):
            return input_shape
        batch_size = input_shape[0]
        # TODO if is none?
        return ( input_shape[0]//self.num_degs, input_shape[1] )

    def get_output_for(self, input, *args, **kwargs):

        input_shape = self.input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
            print 'input shape none'
        print input_shape

        batch_size2 = input_shape[0]//self.num_degs
        num_features = input_shape[1]

        out_pool_shape = (batch_size2, self.num_degs, num_features)
        if self.method=='max':
            out_pooled = T.max(input.reshape(out_pool_shape), 1)
        elif self.method=='avg':
            out_pooled = T.mean(input.reshape(out_pool_shape), 1)
        else:
            raise Exception('unknown pooling type')

        return out_pooled

class SplitLayer(Layer):
    """
    select k features, starting at ix, out of all features.
    useful to split features into subsets.
    """
    def __init__(self, input_layer, ix, k=1, **kwargs):
        super(SplitLayer, self).__init__(input_layer, **kwargs)
        self.ix = ix
        self.k = k

    def get_output_shape_for(self, input_shape):
        #if any(s is None for s in input_shape):
        #    return input_shape
        return (input_shape[0], self.k,) + input_shape[2:]

    def get_output_for(self, input, *args, **kwargs):
        #input_shape = self.input_shape
        #if any(s is None for s in input_shape):
            #input_shape = input.shape
        return input[:,self.ix:(self.ix+self.k)]
