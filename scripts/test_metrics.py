
import random
import time
import voxnet.metrics_logging

logger = voxnet.metrics_logging.MetricsLogger('foo.mlog', reinitialize=True)
logger.log(foo=32, bar=21)

ctr = 0
while True:
    #print '.',
    logger.log({'foo': '%s'%ctr, 'bar': 32}, bla=random.random())
    #time.sleep(0.1)
    ctr += 1

