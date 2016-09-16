#!/usr/bin/env python
import argparse
import os
import sys

import numpy
from PIL import Image
import six.moves.cPickle as pickle


parser = argparse.ArgumentParser(description='Compute images mean array')
parser.add_argument('dataset', help='Path to training image-label list file')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--output', '-o', default='output.txt',
                    help='path to output mean array')
args = parser.parse_args()

f = open(args.output,'w')

sum_image = None
count = 0
for line in open(args.dataset):
    filepath = os.path.join(args.root, line.strip().split()[0])
    if os.path.exists(filepath) == True:
#        print filepath + ' ' + line.strip().split()[1] +'\n'
        f.write(filepath + ' ' + line.strip().split()[1] +'\n')

f.close()