# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:15:47 2016
"""
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from datetime import datetime
import math
from sklearn.metrics import mean_absolute_error

import os
import time


from time import clock
from PIL import Image

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import six

fDoMode = 0

import sys
import pickle
import numpy as np

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data

X_train = None
y_train = []

for i in range(1,6):
    data_dic = unpickle("cifar-10-batches-py/data_batch_{}".format(i))
    if i == 1:
        X_train = data_dic['data']
    else:
        X_train = np.vstack((X_train, data_dic['data']))
    y_train += data_dic['labels']
X_train = X_train.reshape((len(X_train),3, 32, 32))/255.
y_train = np.array(y_train)

test_data_dic = unpickle("cifar-10-batches-py/test_batch")
X_test = test_data_dic['data']
X_test = X_test.reshape(len(X_test),3,32,32)/255.
y_test = np.array(test_data_dic['labels'])

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print("X_train shape",X_train.shape)
print("y_train shape",y_train.shape)

print('学習データの数:', len(X_train))
print('検証データの数:', len(X_test))

# Network definition
class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(
        conv1=F.Convolution2D(3, 32, 3, pad=1),
        conv2=F.Convolution2D(32, 32, 3, pad=1),
        l1=F.Linear(2048, 1024),
        l2=F.Linear(1024, 10)
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None
    
#    def forward(x_data, y_data, train=True):
    def __call__(self, x):#, t):
#        x, t = chainer.Variable(x), chainer.Variable(t)
#        x = chainer.Variable(x)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), train=self.train)
        y = self.l2(h)
#        print("end __call__")
#        print("y shape",y.data.shape)
        return y
        """
        h = F.reshape(y, (x.data.shape[0], 9))
        print("final shape",h.data.shape)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        return self.loss
        """

        """
        if self.train:
           return F.softmax_cross_entropy(y, t)
        else:
            return F.accuracy(y, t)

        """

"""
    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
"""

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

batchsize = 100
n_epoch = 20

# Prepare multi-layer perceptron model, defined in net.py
#if args.net == 'simple':
model = L.Classifier(CNN())

#    model = L.Classifier(net.MnistMLP(784, n_units, 10))
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(X_train.shape[0])
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, X_train.shape[0], batchsize):
        x = chainer.Variable(xp.asarray(X_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))
#        x = xp.asarray(X_train[perm[i:i + batchsize]])
#        t = xp.asarray(y_train[perm[i:i + batchsize],0])

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)

        if epoch == 1 and i == 0:
            with open('graph.dot', 'w') as o:
                g = computational_graph.build_computational_graph(
                    (model.loss, ), remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / X_train.shape[0], sum_accuracy / X_train.shape[0]))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, X_test.shape[0], batchsize):
        x = chainer.Variable(xp.asarray(X_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                             volatile='on')
#        x = xp.asarray(X_test[i:i + batchsize])
#        t = xp.asarray(y_test[i:i + batchsize,0])
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / X_test.shape[0], sum_accuracy / X_test.shape[0]))

# Save the model and the optimizer
print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)
