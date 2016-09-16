# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:33:15 2016

@author: d2713
"""

import os
from sklearn.cross_validation import train_test_split

dir_path = r'D:\nis_local\pywork\chainer_work\myproj\caltech256\data\256_ObjectCategories'

dir_lists = next(os.walk(dir_path))[1]

X=[]
y=[]
with open("train.txt", "w") as train_file:
    with open("test.txt", "w") as test_file:    
        for i,dir_name in enumerate(dir_lists):
            print(i,dir_name)
            
            files = os.listdir(dir_path + '/' + dir_name)
            for file in files:
                X.append(dir_path + '\\' + dir_name + '\\' + str(file))
                y.append(i)
            
            
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
        for xt,yt in zip(X_train,y_train):
            print(str(xt) + ' ' +  str(yt) + '\n')
            train_file.write(str(xt) + ' ' +  str(yt) + '\n')
    
        for xt,yt in zip(X_test,y_test):
            print(str(xt) + ' ' +  str(yt) + '\n')
            test_file.write(str(xt) + ' ' +  str(yt) + '\n')

'''
with open("train.txt", "w") as train_file:
    with open("test.txt", "w") as test_file:    
        for i,dir_name in enumerate(dir_lists):
            print(i,dir_name)
            
            files = os.listdir(dir_path + '/' + dir_name)
            X_train, X_test, y_train, y_test = train_test_split(files, [i] * len(files), test_size=0.33, random_state=42)
            for t in X_train:
                train_file.write(dir_path + '\\' + dir_name + '\\' + str(t) + ' ' +  str(i) + '\n')
    
            for t in X_test:
                test_file.write(dir_path + '\\' + dir_name + '\\' + str(t) + ' ' +  str(i) + '\n')
'''    