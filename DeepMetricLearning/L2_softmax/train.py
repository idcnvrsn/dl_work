#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:59:47 2019

"""

from glob import glob
import numpy as np
from keras.utils import to_categorical
import imageio
import cv2
import keras
from keras.applications import MobileNetV2
from keras.applications.xception import Xception
from keras.models import Model
from keras import backend as K
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import argparse
import os
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm
from pprint import pprint

def train(x, y, classes, val_x ,val_y,epoch,batch_size):
    print("L2-SoftmaxLoss training...")

    base_model = MobileNetV2(include_top=False, input_shape=x.shape[1:], alpha=0.5, weights='imagenet')
#    base_mobile = Xception(include_top=True, input_shape=x.shape[1:],weights='imagenet')
#    base_mobile = NASNetLarge(include_top=True, input_shape=x.shape[1:],weights='imagenet')
    model = Model(inputs=base_model.input, outputs=keras.layers.GlobalAveragePooling2D()(base_model.output))

    alpha = 5.0
    l2_softmax_layers = keras.layers.Lambda(lambda x: alpha*(x)/K.sqrt(K.sum(x**2)) ,name='l2_normalization_and_scale')(model.output)
    l2_softmax_layers = Dense(classes, activation='softmax')(l2_softmax_layers)
    model = Model(inputs=model.input, outputs=l2_softmax_layers)

    #model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, amsgrad=True),
                  metrics=['accuracy'])

    #学習
    hist = model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose = 1, validation_data=(val_x,val_y))

    return model

def align(image, size):
    w = image.shape[1]
    h = image.shape[0]
    aligned_image = np.zeros((size,size,image.shape[2]), dtype=np.uint8)
    print(image.shape)
    aligned_image[0:h,0:w,:]=image
    return aligned_image

def save_images(images, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print("save " + str(images.shape[0]) + " images to " + output_dir)
    for index, image in enumerate(images):
        imageio.imwrite(output_dir+os.sep+str(index)+".jpg", image)

def crop(image,x,y,w,h,image_size):
    half_w = int(w/2)
    half_h = int(h/2)
    
    cx = x+half_w
    cy = y+half_h

    lt_x = cx - int(image_size/2)
    if lt_x < 0:
        lt_x = 0
    lt_y = cy - int(image_size/2)
    if lt_y < 0:
        lt_y = 0
    rb_x = cx + int(image_size/2)
    if rb_x > image.shape[1]:
        rb_x =image.shape[1]
    rb_y = cy + int(image_size/2)
    if rb_y > image.shape[0]:
        rb_y =image.shape[0]

    return image[lt_y:rb_y,lt_x:rb_x,:]

def to3ch(image):
    image_ = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    image_[:,:,0] = image
    image_[:,:,1] = image
    image_[:,:,2] = image
    return image_

def load_from_dir(image_dirs,image_size):
    t = []
    freqs = []
    
    for dir_index, image_dir in enumerate(image_dirs):
        image_filenames = glob(image_dir+os.sep+"*.jpg")
        image_filenames = image_filenames#[:1000]
    
        normal_images = np.zeros((len(image_filenames),image_size,image_size,3),dtype=np.uint8)
        for index, image_filename in enumerate(image_filenames):
            image = imageio.imread(image_filename, as_gray=False, pilmode="RGB")
            image = cv2.resize(image,(image_size,image_size))
            normal_images[index] = image
        t.extend([dir_index]*len(image_filenames))
        freqs.append(len(image_filenames))

    return normal_images, t, freqs

def load_from_coco(ids,coco,image_file_size,ann_limit,mscoco_dir):
    t = []
    freqs = []
    normal_images_list = np.empty((0,image_file_size,image_file_size,3), dtype=np.uint8)
    for index, cat_id in enumerate(ids):
        imgIds = coco.getImgIds(catIds=[cat_id] );
        annIds = coco.getAnnIds(imgIds=imgIds, catIds=[cat_id], iscrowd=None)
        if len(annIds) > ann_limit:
            annIds = annIds[:ann_limit]
        anns = coco.loadAnns(annIds)
        normal_images = np.zeros((len(anns),image_file_size,image_file_size,3),dtype=np.uint8)

        for ann_index,ann in enumerate(tqdm(anns)):
            img = coco.loadImgs([ann["image_id"]])[0]
            image = imageio.imread(mscoco_dir+os.sep+img['file_name'])
            if len(image.shape) == 2:
                image = to3ch(image)
            
            x=int(ann['bbox'][0])
            y=int(ann['bbox'][1])
            w=int(ann['bbox'][2])
            h=int(ann['bbox'][3])
            
            crop_image = crop(image,x,y,w,h,args.image_size)
            
            crop_image_norm = np.zeros((image_file_size,image_file_size,image.shape[2]), dtype=np.uint8)                    
            crop_image_norm[0:crop_image.shape[0],0:crop_image.shape[1],:] = crop_image
            
            normal_images[ann_index] = crop_image_norm
            
        t.extend([index]*len(anns))
        freqs.append(len(anns))

        normal_images_list=np.vstack([normal_images_list,normal_images])
    
    return normal_images_list, t, freqs

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='L2-softmaxを使ったDeep Metric Learning',fromfile_prefix_chars='@')

    parser.add_argument('-n', '--normal_dataset', nargs="*",default=["dir","images_labels_normal/images"], help='正常画像を指定する。冒頭がdirでディレクトリ指定(クラスは１つとして扱う。例 -n dir dataset/image)。冒頭がcocoでcocoのid指定(-n coco 2 6)。')
    parser.add_argument('-a', '--anomaly_dataset', nargs="*",default=["dir","images_labels_ano/images"], help='異常画像を指定する。容量は--normal_datasetと同じ。')
    parser.add_argument('-r', '--ref_dataset', nargs="*",default=["dir","images_labels_ref/images"], help='リファレンス画像を指定する。容量は--normal_datasetと同じ。')
    parser.add_argument('-e', '--epoch', default=30, type=int, help='学習する最大エポック数')
    parser.add_argument('-b', '--batch_size', default=24, type=int, help='学習する最大エポック数')
    parser.add_argument('-s', '--image_size', default=224, type=int, help='学習時バッチサイズ')
    parser.add_argument('--mscoco_dir', default="val2017",help="mscocoデータセットのディレクトリ")
    parser.add_argument('--mscoco_annotations_dir', default="annotations_trainval2017",help="mscocoデータセットのアノテーションディレクトリ")
    parser.add_argument('--pascal_voc_dir', default="VOCdevkit/VOC2012")
    parser.add_argument('--save_img', action='store_true',help='クロップ画像の保存を行う場合')
    parser.add_argument('--ann_limit', default=1000, type=int, help='アノテーションの限界サイズ')
    
    args = parser.parse_args()
    pprint(args.__dict__)

    image_file_size = args.image_size

    # 正常データ読み込み
    if args.normal_dataset[0] == "dir":
        normal_images, y_normal, freqs = load_from_dir(args.normal_dataset[1:],args.image_size)

    elif args.normal_dataset[0] == "coco":

        coco=COCO(args.mscoco_annotations_dir+os.sep+"annotations/instances_val2017.json")
        
        ids = [int(_id) for _id in args.normal_dataset[1:] ]
        normal_images , y_normal, normal_freq = load_from_coco(ids,coco,args.image_size,args.ann_limit,args.mscoco_dir)
        
    if args.save_img:
        save_images(normal_images,"normal_images")

    normal_images = normal_images.astype('float32') / 255
    y_normal = to_categorical(y_normal)


    # リファレンスデータ読み込み
    if args.ref_dataset[0] == "dir":
        ref_images, y_ref, freqs = load_from_dir(args.ref_dataset[1:],args.image_size)

    elif args.ref_dataset[0] == "coco":

        coco=COCO(args.mscoco_annotations_dir+os.sep+"annotations/instances_val2017.json")
        
        ids = [int(_id) for _id in args.ref_dataset[1:] ]
        ref_images , y_ref, ref_freqs = load_from_coco(ids,coco,args.image_size,args.ann_limit,args.mscoco_dir)
        
    if args.save_img:
        save_images(ref_images,"ref_images")

    ref_images = ref_images.astype('float32') / 255
    y_ref = to_categorical(y_ref)

    # テストデータ(異常)読み込み
    if args.anomaly_dataset[0] == "dir":
        ano_images, y_ano, freqs = load_from_dir(args.anomaly_dataset[1:],args.image_size)

    elif args.anomaly_dataset[0] == "coco":

        coco=COCO(args.mscoco_annotations_dir+os.sep+"annotations/instances_val2017.json")
        
        ids = [int(_id) for _id in args.anomaly_dataset[1:] ]
        ano_images , y_ano, ano_freqs = load_from_coco(ids,coco,args.image_size,args.ann_limit,args.mscoco_dir)
        
    if args.save_img:
        save_images(ano_images,"ano_images")

    ano_images = ano_images.astype('float32') / 255
    y_ano = to_categorical(y_ano)

    # 3種類のyについて幅を合わせる
    # normal
    y_zero = np.zeros((y_normal.shape[0],(y_ref.shape[1]+y_ano.shape[1]) ),dtype=np.float32)
    y_normal_ = np.hstack([y_normal,y_zero])

    # ref
    y_zero = np.zeros((y_ref.shape[0],y_normal.shape[1]),dtype=np.float32)
    y_ref_ = np.hstack([y_zero,y_ref])
    y_zero = np.zeros((y_ref_.shape[0],y_ano.shape[1]),dtype=np.float32)
    y_ref_ = np.hstack([y_ref_,y_zero])

    # ano
    y_zero = np.zeros((y_ano.shape[0], (y_normal.shape[1]+y_ref.shape[1]) ),dtype=np.float32)
    y_ano_ = np.hstack([y_zero,y_ano])
    
    y_normal = y_normal_
    y_ref = y_ref_
    y_ano = y_ano_

    # train,val,testに分割する
    normal_train_images, normal_test_images, y_normal_train, y_normal_test = train_test_split(normal_images, y_normal, test_size=0.2, random_state=1)
    normal_train_images, normal_val_images, y_normal_train, y_normal_val = train_test_split(normal_train_images, y_normal_train, test_size=0.2, random_state=1)
    ref_train_images, ref_test_images, y_ref_train, y_ref_test = train_test_split(ref_images, y_ref, test_size=0.2, random_state=1)
    ref_train_images, ref_val_images, y_ref_train, y_ref_val = train_test_split(ref_train_images, y_ref_train, test_size=0.2, random_state=1)
    ano_val_images, ano_test_images, y_ano_val, y_ano_test = train_test_split(ano_images, y_ano, test_size=0.8, random_state=1)

    # テスト画像の保存
    np.save("normal_test_images.npy", normal_test_images)
    np.save("ref_test_images.npy", ref_test_images)
    np.save("ano_test_images.npy", ano_test_images)

    #L2-SoftmaxLoss
    model = train(np.vstack((normal_train_images, ref_train_images)), np.vstack((y_normal_train, y_ref_train)), y_ref_train.shape[1],
                     np.vstack((normal_val_images,ref_val_images)), np.vstack((y_normal_val,y_ref_val)), args.epoch ,args.batch_size)

    model.save("model.hdf5")

    #最終層削除
    model.layers.pop()
    model_ev = Model(inputs=model.input,outputs=model.layers[-1].output)

    model_ev.save("model_ev.hdf5")

    # 次元圧縮してプロット
    print("ploting with dimension reduction...")
    pred_normal_test = model_ev.predict(normal_test_images, batch_size=1)
    pred_ref_test = model_ev.predict(ref_test_images, batch_size=1)
    pred_ano_test = model_ev.predict(ano_test_images, batch_size=1)

    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(np.vstack((pred_normal_test,pred_ref_test,pred_ano_test)))
    print(X_reduced.shape)

    range_normal = range(0,pred_normal_test.shape[0])
    range_ref = range(pred_normal_test.shape[0],pred_normal_test.shape[0]+pred_ref_test.shape[0])
    range_ano = range(pred_normal_test.shape[0]+pred_ref_test.shape[0],pred_normal_test.shape[0]+pred_ref_test.shape[0]+pred_ano_test.shape[0])

    plt.scatter(X_reduced[range_normal, 0], X_reduced[range_normal, 1], s=3, c="blue", label="normal")
    plt.scatter(X_reduced[range_ref, 0], X_reduced[range_ref, 1], s=3, c="green", label="reference")
    plt.scatter(X_reduced[range_ano, 0], X_reduced[range_ano, 1], s=3, c="red", label="anomaly")
    plt.legend(loc='best')
#    target = np.hstack(([0]*pred_normal_test.shape[0],[1]*pred_ref_test.shape[0],[2]*pred_ano_test.shape[0]))
#    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=3,c=target)

    plt.savefig("distribution.png")

