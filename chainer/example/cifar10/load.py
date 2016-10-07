# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:13:13 2016
"""
import joblib
import os

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

os.listdir("cifar-10-batches-py")

files = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']

for f in files:
    d = unpickle("cifar-10-batches-py/data_batch_1")
    joblib.dump(d,f + ".pkl")

#a = joblib.load("d.pkl")

