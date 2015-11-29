
import cStringIO as StringIO
import argparse
import imp
import time
import matplotlib
matplotlib.use('Agg')
import seaborn
seaborn.set_style('whitegrid')

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from path import Path

from matplotlib import rcParams

import voxnet

def set_matplotlib_params():
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    golden_ratio = (np.sqrt(5)-1.0)/2.0
    fig_width_in = 8.
    fig_height_in = fig_width_in * golden_ratio
    rcParams['figure.figsize'] = [fig_width_in, fig_height_in]


parser = argparse.ArgumentParser()
parser.add_argument('metrics_fname', type=Path)
parser.add_argument('out_fname', type=Path)
args = parser.parse_args()

set_matplotlib_params()

recs = list(voxnet.metrics_logging.read_records(args.metrics_fname))

#stamps = [np.datetime64(r['_stamp']) for r in recs]
stamps = [r['_stamp'] for r in recs]
df = pd.DataFrame(recs, index=stamps)
df['loss'] = df['loss'].astype(np.float)
#df = df.sort_index()

#loss = df['loss'].sort_index()
df_nonan = df.dropna(axis=0)
acc = df_nonan['acc'].sort_index()
acc.index = df_nonan['itr'].sort_index()

loss = df['loss'].sort_index()
loss.index = df.sort_index()['itr']

smoothing_window = 32

with open(args.out_fname, 'w') as page:
    page.write('<html><head></head><body>')
    page.write('<h1>Training report</h1>')
    page.write('<p>{}</p>'.format(time.ctime()))
    page.write('<h2>Loss</h2>')
    fig = pl.figure()
    loss.plot(label='raw')
    pd.rolling_mean(loss, smoothing_window).plot(label='smoothed')
    pl.xlabel('Iter')
    pl.ylabel('Loss')
    pl.legend()
    fig.tight_layout(pad=0.1)
    sio = StringIO.StringIO()
    pl.savefig(sio, format='svg')
    page.write(sio.getvalue())

    page.write('<h2>Accuracy</h2>')
    fig = pl.figure()
    acc.plot(label='raw')
    pd.rolling_mean(acc, smoothing_window).plot(label='smoothed')
    pl.xlabel('Iter')
    pl.ylabel('Accuracy')
    fig.tight_layout(pad=0.1)
    sio = StringIO.StringIO()
    pl.savefig(sio, format='svg')
    page.write(sio.getvalue())
    page.write('</body></html>')
