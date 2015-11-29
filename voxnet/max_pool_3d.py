"""
max_pool_3d is from https://github.com/lpigou/Theano-3D-ConvNet,
with a "public-domain" license, shown below.
"""

"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org>
"""

import theano.tensor as T

def max_pool_3d(input, ds, ignore_border=False):
    """
    Takes as input a N-D tensor, where N >= 3. It downscales the input video by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1],ds[2]) (time, height, width)

    :type input: N-D theano tensor of input images.
    :param input: input images. Max pooling will be done over the 3 last dimensions.
    :type ds: tuple of length 3
    :param ds: factor by which to downscale. (2,2,2) will halve the video in each dimension.
    :param ignore_border: boolean value. When True, (5,5,5) input with ds=(2,2,2) will generate a
      (2,2,2) output. (3,3,3) otherwise.
    """

    if input.ndim < 3:
        raise NotImplementedError('max_pool_3d requires a dimension >= 3')

    # extract nr dimensions
    vid_dim = input.ndim
    # max pool in two different steps, so we can use the 2d implementation of
    # downsamplefactormax. First maxpool frames as usual.
    # Then maxpool the time dimension. Shift the time dimension to the third
    # position, so rows and cols are in the back

    # extract dimensions
    frame_shape = input.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input.shape[:-2])
    batch_size = T.shape_padright(batch_size, 1)

    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = T.cast(T.join(0, batch_size, T.as_tensor([1, ]), frame_shape),
                       'int32')
    input_4D = T.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of videos in rows and cols
    op = T.signal.downsample.DownsampleFactorMax((ds[1], ds[2]), ignore_border)
    output = op(input_4D)
    # restore to original shape
    outshape = T.join(0, input.shape[:-2], output.shape[-2:])
    out = T.reshape(output, outshape, ndim=input.ndim)

    # now maxpool time

    # output (time, rows, cols), reshape so that time is in the back
    shufl = (list(range(vid_dim - 3)) + [vid_dim - 2] + [vid_dim - 1] +
             [vid_dim - 3])
    input_time = out.dimshuffle(shufl)
    # reset dimensions
    vid_shape = input_time.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = T.prod(input_time.shape[:-2])
    batch_size = T.shape_padright(batch_size, 1)

    # store as 4D tensor with shape: (batch_size,1,width,time)
    new_shape = T.cast(T.join(0, batch_size, T.as_tensor([1, ]), vid_shape),
                       'int32')
    input_4D_time = T.reshape(input_time, new_shape, ndim=4)
    # downsample mini-batch of videos in time
    op = T.signal.downsample.DownsampleFactorMax((1, ds[0]), ignore_border)
    outtime = op(input_4D_time)
    # output
    # restore to original shape (xxx, rows, cols, time)
    outshape = T.join(0, input_time.shape[:-2], outtime.shape[-2:])
    shufl = (list(range(vid_dim - 3)) + [vid_dim - 1] + [vid_dim - 3] +
             [vid_dim - 2])
    return T.reshape(outtime, outshape, ndim=input.ndim).dimshuffle(shufl)
