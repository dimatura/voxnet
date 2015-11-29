
#import ipdb
import numpy as np
import matplotlib.pyplot as pl
import seaborn
import pandas as pd

def plot_metrics(records, keys, styles=None, figsize=None):
    """
    TODO smoothing not implemented yet
    TODO styles, figsize
    """
    #ipdb.set_trace()
    if len(records)==0:
        return None

    df = pd.DataFrame(records)
    for k in keys:
        df[k] = df[k].astype(np.float32)

    #fig, axes = pl.subplots(nrows=len(keys), ncols=1)
    #(pd.stats.moments.rolling_mean(train_df.acc.dropna().tail(tail), window)).plot()
    fig = pl.figure()
    if keys is None:
        df.plot(subplots=True)
    else:
        keys = list(keys)
        df[keys].plot(subplots=True)
    pl.xlabel('iteration')
    #valid_df.acc.plot('line')
    return fig

"""
# pl.subplot
pl.subplot(4, 1, 1)
(pd.stats.moments.rolling_mean(train_df.acc.dropna().tail(tail), window)).plot()
pl.xlabel('iteration')
pl.ylabel('accuracy')
pl.subplot(4, 1, 2)
pl.subplot(4, 1, 3)
valid_df.avg_fscore.dropna().plot()
pl.xlabel('iteration')
pl.ylabel('avg f1')
pl.subplot(4, 1, 3)
valid_df.avgw_fscore.dropna().plot()
pl.xlabel('iteration')
pl.ylabel('avgw f1')
pl.legend()
pl.subplot(4, 1, 4)
(pd.stats.moments.rolling_mean(train_df.loss.tail(tail), window)).plot()
pl.xlabel('iteration')
pl.ylabel('loss')
"""
