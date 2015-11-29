
import voxnet
import numpy as np

N = 10

wr = voxnet.io.NpyTarWriter('foo.tar')
for i in xrange(N):
    x = np.random.random((32,)*3)
    wr.add(x, '%03d' % i)
wr.close()

s = 0.
rd = voxnet.io.NpyTarReader('foo.tar')
for name, x in rd:
    print name
    s += x.sum()
print s/N
