import os
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import cv2
import numpy as np
import keras
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
import argparse

from imageio import imread
from skimage.transform import resize

import joblib

def transform_y(y_pred, scaler_test):

    y_pred = y_pred.reshape(-1, 1)

    y_pred = scaler_test.inverse_transform(y_pred)
    y_pred = y_pred.flatten()
    return y_pred

def predict(model, scaler, file_name):
    img_height = 512
    img_width = 512
    num_channels = 3
    batch_size = 1

    # scaler = joblib.load('waterquality_Flask/MaxAbsScaler.pkl')
    # model = load_model("waterquality_Flask/weights_mobileNet_without_pre1570856691.343963_color.h5")

    data_path = file_name
    # data_path = data_path.split('/')[-1]

    X = np.zeros((1, img_height, img_width, num_channels), dtype=np.float32)
    Z = np.zeros((1, img_height, img_width, num_channels), dtype=np.float32)
    y = np.ones(1, dtype=np.float32)

    # load as RGB
    #img = imread(data_path)
    img = imread(data_path, as_gray=False, pilmode="RGB")
    # # SKLearn resize
    img = resize(img, (img_height, img_width))
    img = np.array(img)
    X[0] = img


    y_pred = model.predict(X, verbose=1)
    y_pred = transform_y(y_pred, scaler)
    return y_pred[0]




