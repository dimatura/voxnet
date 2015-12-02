# voxnet


3D/Volumetric Convolutional Neural Networks with Theano+Lasagne.

![example rendering](https://github.com/dimatura/voxnet/blob/master/doc/instance.png)

## Installation

`voxnet` is based on [Theano](http://deeplearning.net/software/theano/) 
and [Lasagne](http://deeplearning.net/software/theano/).

You will also need [path.py](https://github.com/jaraco/path.py) and
[scikit-learn](http://scikit-learn.org/stable/). Scikit-learn
is used purely for evaluation of accuracy and is an easily removable dependency.

You can do something like
```sh
git clone git@github.com:dimatura/voxnet.git
cd voxnet
pip install --editable .
```


## ModelNet10 Example

### Get data 

In this example we will use the ModelNet 10 dataset, 
from the excellent [3D ShapeNet](http://3dshapenets.cs.princeton.edu/) project.

To make our life easier we will use the voxelized version, which
is included in the source code distribution. Unfortunately,
it comes in evil `.mat` files, so we will convert them to a 
more python-friendly data format first.

`scripts/download_shapenet10.sh` will try to download and convert the data
for you. This may take a while.

```sh
# scripts/download_shapenet10.sh
wget http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip 
unzip 3DShapeNetsCode
python convert_shapenet10.py 3DShapeNets
```

If you're curious, the data format is simply a `tar` file
consisting of zlib-compressed `.npy` files. Simple and effective!

### Train model

We will be messy and do everything in the `scripts/` directory.


```sh
cd scripts/
python train.py config/shapenet10.py shapenet10_train.tar
```

`config/shapenet10.py` stores the model architecture
and hyperparameters related to training as Python code.
`train.py` loads this code dynamically, compiles the Theano model,
and begins training with the data from `shapenet10_train.tar`.
Note that compiling the Theano model might around a minute
for the first execution.
As soon as training begins, metrics will be printed to stdout and
learned weights are periodically saved to `weights.npz`.

During training (which will take a few hours) you can monitor progress
visually by by running `scripts/train_reports.py`. Note that this script has a
few dependencies, including `seaborn` and `pandas`.  The script uses training
metrics stored in a file called `metrics.jsonl`.  This is simply a format with
one `json` record per line (inspired by [JSON Lines](http://jsonlines.org)).


### Test model

```sh
python test.py config/shapenet10.py shapenet10_test.tar --out-fname out.npz
```

`test.py` uses the same model as `train.py`, but only
for classifying instances from the test set. It performs
simple evaluation and optionally saves the predictions in an 
`.npz` file.

If you don't want to train your own, you can use the
`--weights` option with the
an example result of running this script, 
`shapenet10_weights.npz`. This file was committed with
[Git LFS](https://git-lfs.github.com/), so you
can use that, or simply download the raw version 
from github. 


### Visualize


If you have CPU cycles to burn, try

```sh
python output_viz.py out.npz shapenet10_test.tar out.html
```

This will randomly select 10 instances from the test set,
render them with a very very inefficient renderer, and create
a small page called `out.html` with the renders,
the ground truth and the predicted label (see example above).
Requires [gizeh](https://github.com/Zulko/gizeh).

## Reference

```
@inproceedings{maturana_iros_2015,
    author = "Maturana, D. and Scherer, S.",
    title = "{VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition}",
    booktitle = "{IROS}",
    year = "2015",
    pdf = "/extra/voxnet_maturana_scherer_iros15.pdf",
}
```

## TODO 

* Cleaning up
* Testing
* More options
* Better visualization
* Use new cudnn 3D capabilities


