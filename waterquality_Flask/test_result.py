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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image (i.e., path of image)")
args = vars(ap.parse_args())

img_height = 512
img_width = 512
num_channels = 3
batch_size = 1

print('-'*30)
print('[INFO] Loading and preprocessing image data...')
print('-'*30)

scaler = joblib.load('MaxAbsScaler.pkl')

data_path = args["image"]
data_path = data_path.split('/')[-1]

X = np.zeros((1, img_height, img_width, num_channels), dtype=np.float32)
Z = np.zeros((1, img_height, img_width, num_channels), dtype=np.float32)
y = np.ones(1, dtype=np.float32)

print("[INFO] Image: {}".format(data_path))

# load as RGB
img = imread(data_path)
# # SKLearn resize
img = resize(img, (img_height, img_width))
img = np.array(img)
X[0] = img

img2 = cv2.imread(data_path)
# # SKLearn resize
img2 = cv2.resize(img2, (img_height, img_width), interpolation=cv2.INTER_LINEAR)
img2 = np.array(img2)
Z[0] = img2

print("[INFO] iamge data from sklearn:")
print(img)

print("[INFO] iamge data from cv2:")
print(img2)

print('-'*30)
print('[INFO] Loading saved weights...')
print('-'*30)

model = load_model("weights_mobileNet_without_pre1570856691.343963_color.h5")

print('-'*30)
print('[INFO] Predicting masks on test data...')
print('-'*30)

y_pred = model.predict(X, verbose=1)
#K.clear_session()

print("[RESULT] Value = {}".format(y_pred))
# y_pred = y_pred * 200.0
y_pred = transform_y(y_pred, scaler)
print("[RESULT] Value = {}".format(y_pred))
