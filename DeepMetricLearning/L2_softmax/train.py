#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:59:47 2019

@author: d2713
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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='L2-softmaxを使ったDeep Metric Learning')

    parser.add_argument('-n', '--normal_image_dir', default="images_labels_normal/images", help='正常画像が保存されたディレクトリ')
    parser.add_argument('-a', '--anomaly_image_dir', default="images_labels_ano/images", help='異常画像が保存されたディレクトリ')
    parser.add_argument('-e', '--epoch', default=30, type=int, help='学習する最大エポック数')
    parser.add_argument('-s', '--image_size', default=224, type=int, help='画像サイズ')
    parser.add_argument('-r', '--refname', default="pascal_voc", help='リファレンスデータ')
    
    args = parser.parse_args()
    print(args)

    image_file_size = args.image_size

    # 正常データ読み込み
    image_filenames = glob(args.normal_image_dir+os.sep+"*.jpg")
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

    # リファレンスデータ読み込み
    if args.refname == "pascal_voc":
        pv_label_filenames = sorted(glob("../../models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Main/*_train.txt"))
        
        y_pv = np.zeros((5717,21),dtype=np.uint8)
        for index, pv_label_filename in enumerate(pv_label_filenames):
            df = pd.read_csv(pv_label_filename,header=None, delim_whitespace=True)
            y_pv[df.iloc[:,1]==1, index] = 1
        
        pv_filenames=["../../models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/" + v[0] + ".jpg" for (k, v) in df.iterrows()]

        pv_filenames=pv_filenames[:1000]
        y_pv = y_pv[:len(pv_filenames)]
    
        pv_images = np.zeros((len(pv_filenames),image_file_size,image_file_size,3),dtype=np.uint8)
        for index, pv_filename in enumerate(pv_filenames):
            pv_image = imageio.imread(pv_filename)
            pv_image = cv2.resize(pv_image,(image_file_size,image_file_size))
            pv_images[index] = pv_image
        
        pv_images = pv_images.astype('float32') / 255
        pv_train_images, pv_test_images, y_pv_train, y_pv_test = train_test_split(pv_images, y_pv, test_size=0.2, random_state=1)
        pv_train_images, pv_val_images, y_pv_train, y_pv_val = train_test_split(pv_train_images, y_pv_train, test_size=0.2, random_state=1)
    else:
        assert False, "No reference is indicated."
    
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
    np.save("pv_test_images.npy", pv_test_images)
    np.save("ano_test_images.npy", ano_test_images)

    #L2-SoftmaxLoss
    model = train_L2(np.vstack((normal_train_images, pv_train_images)), np.vstack((y_normal_train, y_pv_train)), y_pv_train.shape[1],
                     np.vstack((normal_val_images,pv_val_images)), np.vstack((y_normal_val,y_pv_val)), args.epoch )

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
    pred_pv_test = model_ev.predict(pv_test_images, batch_size=1)
    pred_ano_test = model_ev.predict(ano_test_images, batch_size=1)

    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(np.vstack((pred_normal_test,pred_pv_test,pred_ano_test)))
    print(X_reduced.shape)

    range_normal = range(0,pred_normal_test.shape[0])
    range_pv = range(pred_normal_test.shape[0],pred_normal_test.shape[0]+pred_pv_test.shape[0])
    range_ano = range(pred_normal_test.shape[0]+pred_pv_test.shape[0],pred_normal_test.shape[0]+pred_pv_test.shape[0]+pred_ano_test.shape[0])

    plt.scatter(X_reduced[range_normal, 0], X_reduced[range_normal, 1], s=3, c="blue", label="normal")
    plt.scatter(X_reduced[range_pv, 0], X_reduced[range_pv, 1], s=3, c="green", label="reference")
    plt.scatter(X_reduced[range_ano, 0], X_reduced[range_ano, 1], s=3, c="red", label="anomaly")
    plt.legend(loc='best')
#    target = np.hstack(([0]*pred_normal_test.shape[0],[1]*pred_pv_test.shape[0],[2]*pred_ano_test.shape[0]))
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