
from PIL import Image
import numpy as np
import voxnet
from voxnet import isovox

reader = voxnet.io.NpyTarReader('/home/dmaturan/repos/voxnet/scripts/shapenet10_test.tar')
iv = isovox.IsoVox(300, 300, 6.)
iv.yaw = np.pi/6.
iv.stroke_width = 0
#iv.bg_color = (.4, .4, .4)
iv.bg_color = (1.0, 1.0, 1.0)
iv.origin_y = 1.1*iv.height
#img = iv.render(xd[::-1,:,:])
for ix, (xd, yd) in enumerate(reader):
    if ix%12 != 0:
        continue
    img = Image.fromarray(iv.render(xd))
    img.save('test_{:05d}.png'.format(ix))

