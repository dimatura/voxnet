
import numpy as np

import lasagne
import lasagne.layers

import voxnet

lr_schedule = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }

cfg = {'batch_size' : 32,
       'learning_rate' : lr_schedule,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 10,
       'batches_per_chunk': 64,
       'max_epochs' : 80,
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 12,
       'checkpoint_every_nth' : 4000,
       }

def get_model():
    dims, n_channels, n_classes = tuple(cfg['dims']), cfg['n_channels'], cfg['n_classes']
    shape = (None, n_channels)+dims

    l_in = lasagne.layers.InputLayer(shape=shape)
    l_conv1 = voxnet.layers.Conv3dMMLayer(
            input_layer = l_in,
            num_filters = 32,
            filter_size = [5,5,5],
            border_mode = 'valid',
            strides = [2,2,2],
            W = voxnet.init.Prelu(),
            nonlinearity = voxnet.activations.leaky_relu_01,
            name =  'conv1'
        )
    l_drop1 = lasagne.layers.DropoutLayer(
        incoming = l_conv1,
        p = 0.2,
        name = 'drop1'
        )
    l_conv2 = voxnet.layers.Conv3dMMLayer(
        input_layer = l_drop1,
        num_filters = 32,
        filter_size = [3,3,3],
        border_mode = 'valid',
        W = voxnet.init.Prelu(),
        nonlinearity = voxnet.activations.leaky_relu_01,
        name = 'conv2'
        )
    l_pool2 = voxnet.layers.MaxPool3dLayer(
        input_layer = l_conv2,
        pool_shape = [2,2,2],
        name = 'pool2',
        )
    l_drop2 = lasagne.layers.DropoutLayer(
        incoming = l_pool2,
        p = 0.3,
        name = 'drop2',
        )
    l_fc1 = lasagne.layers.DenseLayer(
        incoming = l_drop2,
        num_units = 128,
        W = lasagne.init.Normal(std=0.01),
        name =  'fc1'
        )
    l_drop3 = lasagne.layers.DropoutLayer(
        incoming = l_fc1,
        p = 0.4,
        name = 'drop3',
        )
    l_fc2 = lasagne.layers.DenseLayer(
        incoming = l_drop3,
        num_units = n_classes,
        W = lasagne.init.Normal(std = 0.01),
        nonlinearity = None,
        name = 'fc2'
        )
    return {'l_in':l_in, 'l_out':l_fc2}
