
import numpy as np

def batch_jitter(src, shift_ijk, flip_ijk, fill_val=0):
    if not src.ndim==4: raise Exception('for bct01 data')
    if not len(shift_ijk)==3: raise Exception('shift_ijk must be len 3')
    if not len(flip_ijk)==3: raise Exception('flip_ijk must be len 3')
    dst = np.empty_like(src)
    dst.fill(fill_val)
    steps = [1] # batch
    for flip in flip_ijk:
        steps.append( -1 if flip else 1 )
    slices = [slice(None, None, step) for step in steps]
    dst = src[slices]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            dst = np.roll(dst, shift, axis+1)
    return dst
