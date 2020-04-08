# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
# config=tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.9
# tf.Session(config=config)

import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import pandas as pd
import cv2
import numpy as np
import keras
import csv
from keras.models import Sequential
from keras import optimizers
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import random
import string
from sklearn.model_selection import train_test_split
import sklearn.metrics as m
from sklearn.externals import joblib
# from DProcess import convertRawToXY
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


# from multiNets import load_data_200
import waterquality_Flask.CNN_Regression
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_height = 512
img_width = 512


num_channels = 3 #1

data_batch_size = 8 #8 16 1
smooth = 1.
BATCHSIZE = 1
def train_and_predict( batch_size=1,
        weight_file = 'weights.h5',
        model_file = 'regression_modelVGG.h5',
        X_test_file = 'X_test.npy',
        Y_test_file = 'y_test.npy',
        data_path = '/uploads/168369.jpg'
        ):

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    # scaler = joblib.load('MaxAbsScaler_train.pkl')
    scaler = joblib.load('StandardScaler_train.pkl')
    # X_test = np.load(X_test_file)
    # Y_test = np.load(Y_test_file)
    data_path = data_path
    num_images = 1
    data_path = data_path.split('/')[-1]

    X = np.zeros((num_images, img_height, img_width, num_channels), dtype=np.float32)
    y = np.ones(num_images, dtype=np.float32)

    print(data_path)
    img = cv2.imread(data_path)
    img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_CUBIC)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # img = resize(img, (img_height, img_width))
    img = np.array(img)
    X[0] = img

    # y[0] = float(data_path[:3])

    # y = scaler.fit_transform(y.reshape(-1, 1))
    # y = y.flatten()

    print(X[0])

    # model = get_unet()
    # model = train.CNN(X_train)


    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    # model = load_model('./model_CNN.h5')
    model = load_model(model_file)
    # model.load_weights('./modelWights/'+weight_file)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    # a = a.reshape(-1, 1)
    # imgs_mask_test = model.predict(X_test, batch_size=batch_size, verbose=1)
    # np.save('imgs_mask_test.npy', imgs_mask_test)

    # show_roc_pr_curve( "CNN",
    #                  "12812864_00",
    #                 Y_test,
    #                 imgs_mask_test
    #                 )
    y_pred = model.predict(X, batch_size=batch_size, verbose=1)
    # y_pred = np.load('imgs_mask_test.npy')
    K.clear_session()
    # Y_test = y.reshape(-1, 1)
    # y_pred = y_pred.reshape(-1, 1)

    # # Y_test = scaler.inverse_transform(Y_test)
    # y_pred = scaler.inverse_transform(y_pred)
    print(y_pred)

    y_pred = y_pred * 200.0
    # CNN_Regression.save_result('regress_'+model_file+str(time.time())+'.csv',y_pred,Y_test)
    print(y_pred)
    return y_pred

def compare(test_file = 'y_test_transformed.npy', pred_file = 'y_pred.npy'):
    y_pred = np.load(pred_file)
    y_test = np.load(test_file)
    print(y_pred)
    print(y_test)
    diff = y_pred - y_test
    print(diff)
    print(np.mean(np.absolute(diff)))
    np.savetxt("result.csv", diff, delimiter=",")




def show_roc_pr_curve(model_name = "",
                    results_filename = "metazoa_background_prob_method1_codemode42_",
                    valdata_binary = None, #ground turth
                    results = None  #predict
                    ):

    valdata_binary = np.squeeze(valdata_binary)
    valdata_binary = valdata_binary
    results = results
    # results = np.transpose(results)[0]
    print(valdata_binary)
    print(results)
    print(valdata_binary.shape) #(len)

    print(results.shape) #(-1, 1)
    # results = np.transpose(results)

    fpr, tpr, thresholds = m.roc_curve(valdata_binary, results, pos_label = 1)
    roc_auc = m.auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr,
         lw=lw, label='ROC (auc = %0.2f)' % roc_auc)

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for '+model_name)
    plt.legend(loc="lower right")
    plt.savefig("./result/figures/roc_curve_"+model_name+"_"+results_filename+".png")
    # plt.show()
    plt.clf()

    precision, recall, th = m.precision_recall_curve(valdata_binary, results)
    precision_score_list = []
    roc_auc_score_list = []
    precision_score_list.append(average_precision_score(valdata_binary, results))
    roc_auc_score_list.append(roc_auc_score(valdata_binary, results))

    lw = 2
    plt.plot(recall, precision,
         lw=lw, label='precision_recall' )

    plt.legend(loc="lower right")
    plt.title('precision_recall_curve for '+model_name)
    plt.savefig("./result/figures/precision_recall_curve_"+model_name+"_"+results_filename+".png")
    # plt.show()
    plt.clf()

if __name__ == '__main__':
    # X_train, X_val, y_train, y_val, scaler = None, None, None, None, None
    # X_train, X_val, y_train, y_val, scaler = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images')
    # X_train, X_val, y_train, y_val, scaler = load_data_200( X_train, X_val, y_train, y_val, data_path = './Images2')


    train_and_predict()
    # compare()
