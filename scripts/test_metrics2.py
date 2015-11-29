
import time
import voxnet.metrics_logging


bla = 0.
recs = []
for rec in voxnet.metrics_logging.read_records('foo.mlog'):
    recs.append(rec)
    bla += rec.get('bla', 0.)
print len(recs), bla/len(recs)

#recs = voxnet.metrics_logging.MetricsLogger.read_records2('foo.mlog')
while False:
    recs = list(voxnet.metrics_logging.MetricsLogger.read_records('foo.mlog'))
    print len(recs)
    time.sleep(1)
