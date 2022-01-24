from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
import math
from matplotlib import pyplot as plt
import time
import random
import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
K.set_image_data_format('channels_last')


def expand(x):
    x = K.expand_dims(x, axis=-1)
    return x

def squeeze(x):
    x = K.squeeze(x, axis=-1)
    return x

#3D
def BN_block3d(filter_num, input):
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


#get the croped shape
def get_crop_shape(target, refer):
    # spectral, the 4th dimension
    cs = (target.get_shape()[3] - refer.get_shape()[3])
    assert (cs >= 0)
    if cs % 2 != 0:
        cs1, cs2 = int(cs / 2), int(cs / 2) + 1
    else:
        cs1, cs2 = int(cs / 2), int(cs / 2)
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2),(cs1,cs2)

def Unet_Lightweight(numClass,NC):
    inputs = Input(shape=(96, 96,NC))
    input3d = Lambda(expand)(inputs)
    conv3d1 = BN_block3d(32, input3d)
    pool3d1 = MaxPooling3D(pool_size=2)(conv3d1)

    conv3d2 = BN_block3d(64, pool3d1)

    pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)

    conv3d3 = BN_block3d(128, pool3d2)

    drop3d3 = Dropout(0.3)(conv3d3)

    up3d4 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(drop3d3))
    ch, cw, cs = get_crop_shape(conv3d2, up3d4)
    crop_conv3d2 = Cropping3D(cropping=(ch, cw, cs))(conv3d2)
    merge3d4 = Concatenate()([crop_conv3d2, up3d4])
    conv3d4 = BN_block3d(64, merge3d4)

    up3d5 = Conv3D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(conv3d4))
    ch, cw, cs = get_crop_shape(conv3d1, up3d5)
    crop_conv3d1 = Cropping3D(cropping=(ch, cw, cs))(conv3d1)
    merge3d5 = Concatenate()([crop_conv3d1, up3d5])
    conv3d5 = BN_block3d(32, merge3d5)

    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(conv3d5)
    conv6 = Lambda(squeeze)(x)
    conv7 = Conv2D(numClass, 1, activation='softmax')(conv6)

    model = Model(inputs=[inputs], outputs=[conv7])
    #model.compile(optimizer=Adam(lr=lr), loss=Dice_loss, metrics=[Dice])
    return model

NC = 5
model_Name = "Lightweight"

numClass=6


