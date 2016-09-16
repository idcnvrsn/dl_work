# -*- coding: utf-8 -*-
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time

import numpy as np
from PIL import Image


import six
import six.moves.cPickle as pickle
#import cPickle as pickle
from six.moves import queue

import chainer
import matplotlib.pyplot as plt
import numpy as np
import math
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from matplotlib.ticker import * 
from chainer import serializers
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser(
    description='Image inspection using chainer')
parser.add_argument('txt', help='Path to inspection txt file')
parser.add_argument('--model','-m',default='model', help='Path to model file')
parser.add_argument('--mean', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
args = parser.parse_args()


def read_image(path, center=False, flip=False):
  image = np.asarray(Image.open(path)).transpose(2, 0, 1)
  if center:
    top = left = cropwidth / 2
  else:
    top = random.randint(0, cropwidth - 1)
    left = random.randint(0, cropwidth - 1)
  bottom = model.insize + top
  right = model.insize + left
  image = image[:, top:bottom, left:right].astype(np.float32)
  image -= mean_image[:, top:bottom, left:right]
  image /= 255
  if flip and random.randint(0, 1) == 0:
    return image[:, :, ::-1]
  else:
    return image


mean_image = pickle.load(open(args.mean, 'rb'))

import nin
model = nin.NIN()
#import googlenet
#model = googlenet.GoogLeNet()

#serializers.load_hdf5("gpu1out.h5", model)
serializers.load_npz('model', model)
cropwidth = 256 - model.insize
#model.to_cpu()

from chainer import cuda
xp = cuda.cupy
cuda.get_device(0).use()
model = model.to_gpu()


def predict(net, x):
    h = F.max_pooling_2d(F.relu(net.mlpconv1(x)), 3, stride=2)
    h = F.max_pooling_2d(F.relu(net.mlpconv2(h)), 3, stride=2)
    h = F.max_pooling_2d(F.relu(net.mlpconv3(h)), 3, stride=2)
    h = net.mlpconv4(F.dropout(h, train=net.train))
    h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))
    return F.softmax(h)

#setattr(model, 'predict', predict)

categories = np.loadtxt("labels.txt", str, delimiter="\t")

#test_txt = pd.read_csv(args.txt,delimiter='\t')
test_txt = pd.read_csv(args.txt, sep=" ", header=None,)
y_true = test_txt[1].tolist()
test_txt = test_txt[0]

y_pred = []

for image_file in test_txt:
    print(image_file)
#for image_file in test_txt:
    img = read_image(image_file)
    x = np.ndarray(
            (1, 3, model.insize, model.insize), dtype=np.float32)
    x[0]=img
    x = chainer.Variable(xp.asarray(x), volatile='on')
    
    score = predict(model,x)

    top_k = 1
    result = score.data[0][:101]
    #result = result[:,np.newaxis]
    prediction = [[0 for i in range(2)] for j in range(101)]
    i = 0
    for r,c in zip(result,categories):
    #    print(r,c)
        prediction[i][0] = r
        prediction[i][1] = c
        i += 1
        
    #prediction = np.hstack((result, categories[:,np.newaxis]))
    #prediction = list(prediction)
    pred_sorted = sorted(prediction,key=lambda x:x[0], reverse=True)
    y_pred.append(categories.tolist().index(pred_sorted[0][1]))
    
#    num = pred_sorted[0][1].split(' ')[1]
#    a = num.split('\'')[0]
#    print(int(a))
#    y_pred.append(int(a))

print(accuracy_score(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))

logfilename = 'eval_log.txt'
with open(logfilename, "w") as file:
    file.write('accuracy_score:' + str(accuracy_score(y_true,y_pred)) )
    file.write('\n\n')
    cm = confusion_matrix(y_true,y_pred)
    for r in cm:
        for a in r:
            file.write('%3d' % a)
            file.write(',')
        file.write('\n')
        
