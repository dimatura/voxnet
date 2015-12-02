
import argparse
import imp
import time
import logging

import numpy as np
from path import Path
import theano
import theano.tensor as T
import lasagne

import voxnet
from voxnet import npytar

#import pyvox

def make_training_functions(cfg, model):
    l_out = model['l_out']
    batch_index = T.iscalar('batch_index')
    # bct01
    X = T.TensorType('float32', [False]*5)('X')
    y = T.TensorType('int32', [False]*1)('y')
    out_shape = lasagne.layers.get_output_shape(l_out)
    #log.info('output_shape = {}'.format(out_shape))

    batch_slice = slice(batch_index*cfg['batch_size'], (batch_index+1)*cfg['batch_size'])
    out = lasagne.layers.get_output(l_out, X)
    dout = lasagne.layers.get_output(l_out, X, deterministic=True)

    params = lasagne.layers.get_all_params(l_out)
    l2_norm = lasagne.regularization.regularize_network_params(l_out,
            lasagne.regularization.l2)
    if isinstance(cfg['learning_rate'], dict):
        learning_rate = theano.shared(np.float32(cfg['learning_rate'][0]))
    else:
        learning_rate = theano.shared(np.float32(cfg['learning_rate']))

    softmax_out = T.nnet.softmax( out )
    loss = T.cast(T.mean(T.nnet.categorical_crossentropy(softmax_out, y)), 'float32')
    pred = T.argmax( dout, axis=1 )
    error_rate = T.cast( T.mean( T.neq(pred, y) ), 'float32' )

    reg_loss = loss + cfg['reg']*l2_norm
    updates = lasagne.updates.momentum(reg_loss, params, learning_rate, cfg['momentum'])

    X_shared = lasagne.utils.shared_empty(5, dtype='float32')
    y_shared = lasagne.utils.shared_empty(1, dtype='float32')

    dout_fn = theano.function([X], dout)
    pred_fn = theano.function([X], pred)

    update_iter = theano.function([batch_index], reg_loss,
            updates=updates, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })

    error_rate_fn = theano.function([batch_index], error_rate, givens={
            X: X_shared[batch_slice],
            y: T.cast( y_shared[batch_slice], 'int32'),
        })
    tfuncs = {'update_iter':update_iter,
             'error_rate':error_rate_fn,
             'dout' : dout_fn,
             'pred' : pred_fn,
            }
    tvars = {'X' : X,
             'y' : y,
             'X_shared' : X_shared,
             'y_shared' : y_shared,
             'batch_slice' : batch_slice,
             'batch_index' : batch_index,
             'learning_rate' : learning_rate,
            }
    return tfuncs, tvars

def jitter_chunk(src, cfg):
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+2)
    return dst

def data_loader(cfg, fname):

    dims = cfg['dims']
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            xc = jitter_chunk(xc, cfg)
            yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    if len(yc) > 0:
        # pad to nearest multiple of batch_size
        if len(yc)%cfg['batch_size'] != 0:
            new_size = int(np.ceil(len(yc)/float(cfg['batch_size'])))*cfg['batch_size']
            xc = xc[:new_size]
            xc[len(yc):] = xc[:(new_size-len(yc))]
            yc = yc + yc[:(new_size-len(yc))]

        xc = jitter_chunk(xc, cfg)
        yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))

def main(args):
    config_module = imp.load_source('config', args.config_path)
    cfg = config_module.cfg
    model = config_module.get_model()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(args.metrics_fname))
    mlog = voxnet.metrics_logging.MetricsLogger(args.metrics_fname, reinitialize=True)

    logging.info('Compiling theano functions...')
    tfuncs, tvars = make_training_functions(cfg, model)

    logging.info('Training...')
    itr = 0
    last_checkpoint_itr = 0
    loader = (data_loader(cfg, args.training_fname))
    for epoch in xrange(cfg['max_epochs']):
        loader = (data_loader(cfg, args.training_fname))

        for x_shared, y_shared in loader:
            num_batches = len(x_shared)//cfg['batch_size']
            tvars['X_shared'].set_value(x_shared, borrow=True)
            tvars['y_shared'].set_value(y_shared, borrow=True)
            lvs,accs = [],[]
            for bi in xrange(num_batches):
                lv = tfuncs['update_iter'](bi)
                lvs.append(lv)
                acc = 1.0-tfuncs['error_rate'](bi)
                accs.append(acc)
                itr += 1
            loss, acc = float(np.mean(lvs)), float(np.mean(acc))
            logging.info('epoch: {}, itr: {}, loss: {}, acc: {}'.format(epoch, itr, loss, acc))
            mlog.log(epoch=epoch, itr=itr, loss=loss, acc=acc)

            if isinstance(cfg['learning_rate'], dict) and itr > 0:
                keys = sorted(cfg['learning_rate'].keys())
                new_lr = cfg['learning_rate'][keys[np.searchsorted(keys, itr)-1]]
                lr = np.float32(tvars['learning_rate'].get_value())
                if not np.allclose(lr, new_lr):
                    logging.info('decreasing learning rate from {} to {}'.format(lr, new_lr))
                    tvars['learning_rate'].set_value(np.float32(new_lr))
            if itr-last_checkpoint_itr > cfg['checkpoint_every_nth']:
                voxnet.checkpoints.save_weights('weights.npz', model['l_out'],
                                                {'itr': itr, 'ts': time.time()})
                last_checkpoint_itr = itr


    logging.info('training done')
    voxnet.checkpoints.save_weights('weights.npz', model['l_out'],
                                    {'itr': itr, 'ts': time.time()})

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=Path, help='config .py file')
    parser.add_argument('training_fname', type=Path, help='training .tar file')
    parser.add_argument('--metrics-fname', type=Path, default='metrics.jsonl', help='name of metrics file')
    args = parser.parse_args()
    main(args)
