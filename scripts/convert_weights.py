
import cPickle as pickle
import numpy as np
import h5py

from path import Path
#base_dir = Path('/media/mavscout/dms/results/ModelNet10_orig_db/shapenet10_origdb_nbp_val_vf01')
base_dir = Path('/media/mavscout/dms/results/ModelNet10_orig_db/shapenet10_2_vf01/')
weight_fnames = base_dir.files('weights_*.h5')
weight_fnames = sorted(weight_fnames, key=lambda x: int(x[x.rfind('_')+1:x.rfind('.')]))
input_fname = weight_fnames[-1]

with h5py.File(input_fname, 'r') as f:
    to_save = {}
    to_save['metadata'] = {}
    for k in f.keys():
        if k.endswith('.W') or k.endswith('.b'):
            to_save[k] = f[k].value
        else:
            to_save['metadata'][k] = f[k].value
    to_save['metadata'] = pickle.dumps(to_save['metadata'])

    np.savez_compressed('out.npz', **to_save)
