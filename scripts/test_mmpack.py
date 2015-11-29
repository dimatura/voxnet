
import voxnet
import msgpack

with open('foo.msgpack', 'wb') as f:
    for rec in voxnet.metrics_logging.read_records('foo.mlog'):
        mrec = msgpack.packb(rec, use_bin_type=True)
        f.write(mrec)
