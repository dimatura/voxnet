
import argparse
import logging

from path import Path
import numpy as np

import voxnet
from voxnet import npytar
from voxnet import isovox
from voxnet.data import shapenet10


parser = argparse.ArgumentParser()
parser.add_argument('out_fname', type=Path)
parser.add_argument('test_fname', type=Path)
parser.add_argument('viz_fname', type=Path)
parser.add_argument('--num-instances', type=int, default=10)
args = parser.parse_args()

reader = npytar.NpyTarReader(args.test_fname)
out = np.load(args.out_fname)

yhat = out['yhat']
ygnd = out['ygnd']

iv = isovox.IsoVox()

css = """
html { margin: 0 }
body {
    background:#fff;
    color:#000;
    font:75%/1.5em Helvetica, "DejaVu Sans", "Liberation sans", "Bitstream Vera Sans", sans-serif;
    position:relative;
}
/*dt { font-weight: bold; float: left; clear; left }*/
div { padding: 10px; width: 80%; margin: auto }
img { border: 1px solid #eee }
dl { margin:0 0 1.5em; }
dt { font-weight:700; }
dd { margin-left:1.5em; }
table {
    border-collapse:collapse;
    border-spacing:0;
    margin:0 0 1.5em;
    padding:0;
}
td { padding:0.333em;
    vertical-align:middle;
}
}"""

with open(args.viz_fname, 'w') as f:
    f.write('<html><head><style>')
    f.write(css)
    f.write('</style></head>')
    f.write('<body>')

    to_name = lambda id_: shapenet10.class_id_to_name[str(id_+1)]

    display_ix = 12*np.random.randint(0, len(ygnd), args.num_instances)

    xds, yds = [], []
    for ix, (xd, yd) in enumerate(reader):
        if ix in display_ix:
            dix = ix/12
            img = iv.render(xd, as_html=True)
            f.write('<div>')
            f.write('<table><tr><td>')
            f.write(img)
            f.write('</td>')
            f.write('<td>')
            f.write('<dl><dt>Instance:</dt><dd>{}</dd>'.format(yd))
            f.write('<dt>Predicted label:</dt><dd>{}</dd>'.format(to_name(yhat[dix])))
            f.write('<dt>True label:</dt><dd>{}</dd></dl>'.format(to_name(ygnd[dix])))
            f.write('</td></tr></table>')
            f.write('</div>')
            xds.append(xd)
            yds.append(yd)

    f.write('</body></html>')
