# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("source_dir")
parser.add_argument("target_dir")
args = parser.parse_args()

target_shape = (256, 256)

dir_lists = next(os.walk(args.source_dir))[1]

for i,dir_name in enumerate(dir_lists):
    for source_imgpath in os.listdir(args.source_dir + '/' + dir_name):
        print(dir_name + '\\' + source_imgpath)
        src = cv2.imread(args.source_dir+"/"+ dir_name + '/' + source_imgpath)
        print((args.source_dir+"/"+source_imgpath))
        # mirror image
        while src.shape[0] < target_shape[0] or src.shape[1] < target_shape[1]:
            print(src.shape)
            src = numpy.concatenate((numpy.fliplr(src), src), axis=1)
            src = numpy.concatenate((numpy.flipud(src), src), axis=0)
        src = src[:target_shape[0], :target_shape[1]]
        print(src.shape)
        print(args.target_dir+"/"+ dir_name + '/' + source_imgpath)
        cv2.imwrite(args.target_dir+"/"+ dir_name + '/' + source_imgpath, src)
