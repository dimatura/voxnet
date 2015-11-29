
import cStringIO as StringIO
import tarfile
import time
import zlib

import numpy as np

PREFIX = 'data/'
SUFFIX = '.npy.z'

class NpyTarWriter(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'w|')

    def add(self, arr, name):

        sio = StringIO.StringIO()
        np.save(sio, arr)
        zbuf = zlib.compress(sio.getvalue())
        sio.close()

        zsio = StringIO.StringIO(zbuf)
        tinfo = tarfile.TarInfo('{}{}{}'.format(PREFIX, name, SUFFIX))
        tinfo.size = len(zbuf)
        tinfo.mtime = time.time()
        zsio.seek(0)
        self.tfile.addfile(tinfo, zsio)
        zsio.close()

    def close(self):
        self.tfile.close()


class NpyTarReader(object):
    def __init__(self, fname):
        self.tfile = tarfile.open(fname, 'r|')

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        entry = self.tfile.next()
        if entry is None:
            raise StopIteration()
        name = entry.name[len(PREFIX):-len(SUFFIX)]
        fileobj = self.tfile.extractfile(entry)
        buf = zlib.decompress(fileobj.read())
        arr = np.load(StringIO.StringIO(buf))
        return arr, name

    def close(self):
        self.tfile.close()

