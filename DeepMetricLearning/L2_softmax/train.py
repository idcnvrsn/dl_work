#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:59:47 2019

"""

from glob import glob
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import imageio
import cv2
import keras
from keras.applications import MobileNetV2
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras import backend as K
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from time import time
import argparse
import os
import shutil
import json
from pycocotools.coco import COCO
from tqdm import tqdm

def train_L2(x, y, classes, val_x ,val_y,epoch):
    print("L2-SoftmaxLoss training...")
    mobile = MobileNetV2(include_top=True, input_shape=x.shape[1:], alpha=0.5,
                         weights='imagenet')
    
    # 最終層削除
    mobile.layers.pop()
    model = Model(inputs=mobile.input,outputs=mobile.layers[-1].output)
            
    # L2層と全結合層を付ける
    c = keras.layers.Lambda(lambda xx: 5*(xx)/K.sqrt(K.sum(xx**2)))(model.output) #metric learning
    c = Dense(classes, activation='softmax')(c)
    model = Model(inputs=model.input,outputs=c)

    #model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, amsgrad=True),
                  metrics=['accuracy'])

    #学習
    hist = model.fit(x, y, batch_size=24, epochs=epoch, verbose = 1, validation_data=(val_x,val_y))

    #plt.figure()               
    #plt.plot(hist.history['acc'],label="train_acc")
    #plt.legend(loc="lower right")
    #plt.show()

    return model

def get_auc(Z1, Z2):
    y_true = np.zeros(len(Z1)+len(Z2))
    y_true[len(Z1):] = 1#0:正常、1：異常

    # FPR, TPR(, しきい値) を算出
    fpr, tpr, _ = metrics.roc_curve(y_true, np.hstack((Z1, Z2)))

    # AUC
    auc = metrics.auc(fpr, tpr)

    return fpr, tpr, auc

def auc(Z1_arc, Z2_arc):#, Z1_doc, Z2_doc, Z1_L2, Z2_L2):
    fpr_arc, tpr_arc, auc_arc = get_auc(Z1_arc, Z2_arc)
#    fpr_doc, tpr_doc, auc_doc = get_auc(Z1_doc, Z2_doc)
#    fpr_L2, tpr_L2, auc_L2 = get_auc(Z1_L2, Z2_L2)
    
    # ROC曲線をプロット
    plt.plot(fpr_arc, tpr_arc, label='AUC = %.2f'%auc_arc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()
    
def get_score_doc(model, x_train_normal, x_test_normal, x_test_anomaly):
    train = model.predict(x_train_normal, batch_size=1)
    test_s = model.predict(x_test_normal, batch_size=1)
    test_b = model.predict(x_test_anomaly, batch_size=1)

    train = train.reshape((len(train),-1))
    test_s = test_s.reshape((len(test_s),-1))
    test_b = test_b.reshape((len(test_b),-1))

    ms = MinMaxScaler()
    train = ms.fit_transform(train)
    test_s = ms.transform(test_s)
    test_b = ms.transform(test_b)

    # fit the model
    clf = LocalOutlierFactor(n_neighbors=5)
    y_pred = clf.fit(train[:1000])

    # plot the level sets of the decision function
    Z1 = -clf._decision_function(test_s)
    Z2 = -clf._decision_function(test_b)

    return Z1, Z2

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
    
    for dir_index, image_dir in image_dirs:
        image_filenames = glob(image_dir+os.sep+"*.jpg")
        image_filenames = image_filenames#[:1000]
    
        normal_images = np.zeros((len(image_filenames),image_size,image_size,3),dtype=np.uint8)
        for index, image_filename in enumerate(image_filenames):
            image = imageio.imread(image_filename, as_gray=False, pilmode="RGB")
            image = cv2.resize(image,(image_size,image_size))
            normal_images[index] = image
        t.extend([dir_index]*len(image_filenames))

    return normal_images, t

def load_from_coco(ids,coco,image_file_size,ann_limit,mscoco_dir):
    t = []
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

        normal_images_list=np.vstack([normal_images_list,normal_images])
    
    return normal_images_list, t

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='L2-softmaxを使ったDeep Metric Learning',fromfile_prefix_chars='@')

    parser.add_argument('-n', '--normal_dataset', nargs="*",default=["dir","images_labels_normal/images"], help='正常画像を指定する。冒頭がdirでディレクトリ指定(クラスは１つとして扱う。例 -n dir dataset/image)。冒頭がcocoでcocoのid指定(-n coco 2,6)。')
    parser.add_argument('-a', '--anomaly_dataset', nargs="*",default=["dir","images_labels_ano/images"], help='異常画像を指定する。容量は--normal_datasetと同じ。')
    parser.add_argument('-r', '--ref_dataset', nargs="*",default=["dir","images_labels_ref/images"], help='リファレンス画像を指定する。容量は--normal_datasetと同じ。')
    parser.add_argument('-e', '--epoch', default=30, type=int, help='学習する最大エポック数')
    parser.add_argument('-s', '--image_size', default=224, type=int, help='学習時画像サイズ')
    parser.add_argument('--mscoco_dir', default="val2017",help="mscocoデータセットのディレクトリ")
    parser.add_argument('--mscoco_annotations_dir', default="annotations_trainval2017",help="mscocoデータセットのアノテーションディレクトリ")
    parser.add_argument('--pascal_voc_dir', default="VOCdevkit/VOC2012")
    parser.add_argument('-od', '--old_data_mode', action='store_true',help='データセットの指定モードを古い要領で行う場合')
    parser.add_argument('--save_img', action='store_true',help='クロップ画像の保存を行う場合')
    parser.add_argument('--ann_limit', default=1000, type=int, help='アノテーションの限界サイズ')
    
    args = parser.parse_args()
    print(args)

    image_file_size = args.image_size
    
    if args.old_data_mode is False:
        # 正常データ読み込み
        if args.normal_dataset[0] == "dir":
            normal_images, y_normal = load_from_dir(args.normal_dataset[1:],args.image_size)

        elif args.normal_dataset[0] == "coco":
    
            coco=COCO(args.mscoco_annotations_dir+os.sep+"annotations/instances_val2017.json")
            
            ids = [int(_id) for _id in args.normal_dataset[1:] ]
            normal_images , y_normal = load_from_coco(ids,coco,args.image_size,args.ann_limit,args.mscoco_dir)
            
            if args.save_img:
                save_images(normal_images,"normal_images")

        normal_images = normal_images.astype('float32') / 255
        y_normal = to_categorical(y_normal)
    
        normal_train_images, normal_test_images, y_normal_train, y_normal_test = train_test_split(normal_images, y_normal, test_size=0.2, random_state=1)
        normal_train_images, normal_val_images, y_normal_train, y_normal_val = train_test_split(normal_train_images, y_normal_train, test_size=0.2, random_state=1)

        # リファレンスデータ読み込み
        if args.ref_dataset[0] == "dir":
            ref_images, y_ref = load_from_dir(args.ref_dataset[1:],args.image_size)

        elif args.ref_dataset[0] == "coco":
    
            coco=COCO(args.mscoco_annotations_dir+os.sep+"annotations/instances_val2017.json")
            
            ids = [int(_id) for _id in args.ref_dataset[1:] ]
            ref_images , y_ref = load_from_coco(ids,coco,args.image_size,args.ann_limit,args.mscoco_dir)
            
            if args.save_img:
                save_images(ref_images,"ref_images")

            ref_images = ref_images.astype('float32') / 255
            ref_train_images, ref_test_images, y_ref_train, y_ref_test = train_test_split(ref_images, y_ref, test_size=0.2, random_state=1)
            ref_train_images, ref_val_images, y_ref_train, y_ref_val = train_test_split(ref_train_images, y_ref_train, test_size=0.2, random_state=1)

        # テストデータ(異常)読み込み
        if args.anomaly_dataset[0] == "dir":
            ano_images, y_ano = load_from_dir(args.anomaly_dataset[1:],args.image_size)

        elif args.anomaly_dataset[0] == "coco":
    
            coco=COCO(args.mscoco_annotations_dir+os.sep+"annotations/instances_val2017.json")
            
            ids = [int(_id) for _id in args.anomaly_dataset[1:] ]
            ano_images , y_ano = load_from_coco(ids,coco,args.image_size,args.ann_limit,args.mscoco_dir)
            
            if args.save_img:
                save_images(ano_images,"ano_images")

            ano_images = ano_images.astype('float32') / 255
            ano_val_images, ano_test_images, y_ano_val, y_ano_test = train_test_split(ano_images, y_ano, test_size=0.8, random_state=1)
    
    else:
        # 正常データ読み込み
        image_filenames = glob(normal_image_dir+os.sep+"*.jpg")
        image_filenames = image_filenames#[:1000]
    
        normal_images = np.zeros((len(image_filenames),image_file_size,image_file_size,3),dtype=np.uint8)
        for index, image_filename in enumerate(image_filenames):
            image = imageio.imread(image_filename, as_gray=False, pilmode="RGB")
            image = cv2.resize(image,(image_file_size,image_file_size))
            normal_images[index] = image
        normal_images = normal_images.astype('float32') / 255
    
        y_normal = to_categorical([20]*len(normal_images))
    
        normal_train_images, normal_test_images, y_normal_train, y_normal_test = train_test_split(normal_images, y_normal, test_size=0.2, random_state=1)
        normal_train_images, normal_val_images, y_normal_train, y_normal_val = train_test_split(normal_train_images, y_normal_train, test_size=0.2, random_state=1)

        ref_label_filenames = sorted(glob("VOCdevkit/VOC2012/ImageSets/Main/*_train.txt"))
        
        y_ref = np.zeros((5717,21),dtype=np.uint8)
        for index, ref_label_filename in enumerate(ref_label_filenames):
            df = pd.read_csv(ref_label_filename,header=None, delim_whitespace=True)
            y_ref[df.iloc[:,1]==1, index] = 1
        
        ref_filenames=["VOCdevkit/VOC2012/JPEGImages/" + v[0] + ".jpg" for (k, v) in df.iterrows()]

        ref_filenames=ref_filenames[:1000]
        y_ref = y_ref[:len(ref_filenames)]
    
        ref_images = np.zeros((len(ref_filenames),image_file_size,image_file_size,3),dtype=np.uint8)
        for index, ref_filename in enumerate(ref_filenames):
            ref_image = imageio.imread(ref_filename)
            ref_image = cv2.resize(ref_image,(image_file_size,image_file_size))
            ref_images[index] = ref_image
        
        ref_images = ref_images.astype('float32') / 255
        ref_train_images, ref_test_images, y_ref_train, y_ref_test = train_test_split(ref_images, y_ref, test_size=0.2, random_state=1)
        ref_train_images, ref_val_images, y_ref_train, y_ref_val = train_test_split(ref_train_images, y_ref_train, test_size=0.2, random_state=1)
    
        # テストデータ(異常)読み込み
        ano_image_filenames = glob(args.anomaly_image_dir+os.sep+"*.jpg")
        ano_images = np.zeros((len(ano_image_filenames),image_file_size,image_file_size,3),dtype=np.uint8)
        for index, filename in enumerate(ano_image_filenames):
            image = imageio.imread(filename, as_gray=False, pilmode="RGB")
            image = cv2.resize(image,(image_file_size,image_file_size))
            ano_images[index] = image
        
        ano_images = ano_images.astype('float32') / 255
        y_ano = to_categorical([20]*len(ano_images))
        ano_val_images, ano_test_images, y_ano_val, y_ano_test = train_test_split(ano_images, y_ano, test_size=0.8, random_state=1)
        

    # テスト画像の保存
    np.save("normal_test_images.npy", normal_test_images)
    np.save("ref_test_images.npy", ref_test_images)
    np.save("ano_test_images.npy", ano_test_images)

    #L2-SoftmaxLoss
    model = train_L2(np.vstack((normal_train_images, ref_train_images)), np.vstack((y_normal_train, y_ref_train)), y_ref_train.shape[1],
                     np.vstack((normal_val_images,ref_val_images)), np.vstack((y_normal_val,y_ref_val)), args.epoch )

    model.save("model.hdf5")

    #最終層削除
    model.layers.pop()
    model_ev = Model(inputs=model.input,outputs=model.layers[-1].output)
    Z1_L2, Z2_L2 = get_score_doc(model_ev, normal_train_images, normal_test_images, ano_test_images)

    auc(Z1_L2, Z2_L2)

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

    # kNNを使って距離空間で分類器を学習する
    print("kNN Classification on feature space...")

    # kNNで学習する際の特徴量を得るための推論実施
    print("predicting for val...")
    pred_normal_val = model_ev.predict(normal_val_images, batch_size=1)
    pred_ano_val = model_ev.predict(ano_val_images, batch_size=1)

    # valについて正常、異常のクラスラベル作成
    y_val = np.hstack(([1]*y_normal_val.shape[0], [0]*y_ano_val.shape[0]))

    print("kNN Classifier...")
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(np.vstack((pred_normal_val, pred_ano_val)) , y_val)

    # 予測
    y_pred = clf.predict(np.vstack((pred_normal_test, pred_ano_test)))

    # valについて正常、異常のクラスラベル作成
    y_true_test = np.hstack(([1]*y_normal_test.shape[0], [0]*y_ano_test.shape[0]))
    print(confusion_matrix(y_true_test, y_pred))

    # Speed Benchmark
    inference_time = []
    for ano_test_image in ano_test_images[:50]:
        start_time = time()
        features_ano_test_image = model_ev.predict(ano_test_image[np.newaxis,:,:,:], batch_size=1)
        result = clf.predict(features_ano_test_image)
        inference_time.append(time() - start_time)
    
    print("min(sec)", min(inference_time))
    print("max(sec)", max(inference_time))
    print("ave(sec)", np.mean(inference_time))
