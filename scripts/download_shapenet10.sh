#!/bin/sh

# get data
wget http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip 
unzip 3DShapeNetsCode

# convert to our format
python convert_shapenet10.py 3DShapeNets
