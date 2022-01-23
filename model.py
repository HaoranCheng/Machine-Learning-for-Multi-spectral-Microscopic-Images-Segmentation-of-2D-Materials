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

#2D卷积模块
def BN_block(filter_num, input):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


#3D卷积模块
def BN_block3d(filter_num, input):
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

#BN_LSTMCONV
def BN_convLSTM(filter_num, input):
    x = ConvLSTM2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    #print(x.get_shape())
    x = Permute((3, 1, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#    x = Lambda(expand)(x)
#    x = ConvLSTM2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
    x = Lambda(expand)(x)

    return x

#裁剪：使在concatenate前保证encoder和decoder部分形状一致，此处是为了获取裁剪参数
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

#定义单支路U-net
def Unet(numClass,NC):
    inputs = Input(shape=(96, 96, NC))
    conv1 = BN_block(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(64, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(128, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BN_block(256, pool3)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BN_block(512, pool4)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    #print(UpSampling2D(size=(2, 2))(drop5).get_shape())
    #print(up6.get_shape())
    merge6 = Concatenate()([drop4, up6])
    conv6 = BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(64, merge8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(32, merge9)
    conv10 = Conv2D(numClass, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
#定义U-net
def Unet_3D(numClass,NC):
    inputs = Input(shape=(96, 96,NC))
    input3d = Lambda(expand)(inputs)
    conv3d1 = BN_block3d(32, input3d)
    pool3d1 = MaxPooling3D(pool_size=2)(conv3d1)

    conv3d2 = BN_block3d(64, pool3d1)

    pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)

    conv3d3 = BN_block3d(128, pool3d2)

    pool3d3 = MaxPooling3D(pool_size=2)(conv3d3)

    conv3d4 = BN_block3d(256, pool3d3)
    drop3d4 = Dropout(0.3)(conv3d4)
    pool3d4 = MaxPooling3D(pool_size=2)(drop3d4)
    #print(drop3d4.get_shape()[1])
    conv3d5 = BN_block3d(512, pool3d4)
    drop3d5 = Dropout(0.3)(conv3d5)

    up3d6 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(drop3d5))
    ch, cw, cs = get_crop_shape(drop3d4, up3d6)
    crop_drop3d4 = Cropping3D(cropping=(ch, cw, cs))(drop3d4)
    merge3d6 = Concatenate()([crop_drop3d4, up3d6])
    conv3d6 = BN_block3d(256, merge3d6)

    up3d7 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(conv3d6))
    ch, cw, cs = get_crop_shape(conv3d3, up3d7)
    crop_conv3d3 = Cropping3D(cropping=(ch, cw, cs))(conv3d3)
    merge3d7 = Concatenate()([crop_conv3d3, up3d7])
    conv3d7 = BN_block3d(128, merge3d7)

    up3d8 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(conv3d7))
    ch, cw, cs = get_crop_shape(conv3d2, up3d8)
    crop_conv3d2 = Cropping3D(cropping=(ch, cw, cs))(conv3d2)
    merge3d8 = Concatenate()([crop_conv3d2, up3d8])
    conv3d8 = BN_block3d(64, merge3d8)

    up3d9 = Conv3D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(conv3d8))
    ch, cw, cs = get_crop_shape(conv3d1, up3d9)
    crop_conv3d1 = Cropping3D(cropping=(ch, cw, cs))(conv3d1)
    merge3d9 = Concatenate()([crop_conv3d1, up3d9])
    conv3d9 = BN_block3d(32, merge3d9)
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(conv3d9)
    conv9 = Lambda(squeeze)(x)
    conv10 = Conv2D(numClass, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

#定义U-net
def Unet_Shallow(numClass,NC):
    inputs = Input(shape=(96, 96,NC))
    print(inputs.get_shape())
    input3d = Lambda(expand)(inputs)
    conv3d1 = BN_block3d(32, input3d)
    pool3d1 = MaxPooling3D(pool_size=2)(conv3d1)

    conv3d2 = BN_block3d(64, pool3d1)

    pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)

    conv3d3 = BN_block3d(128, pool3d2)

    pool3d3 = MaxPooling3D(pool_size=2)(conv3d3)

    conv3d4 = BN_block3d(256, pool3d3)
    drop3d4 = Dropout(0.3)(conv3d4)


    up3d5 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(drop3d4))
    ch, cw, cs = get_crop_shape(conv3d3, up3d5)
    crop_conv3d3 = Cropping3D(cropping=(ch, cw, cs))(conv3d3)
    merge3d5 = Concatenate()([crop_conv3d3, up3d5])
    conv3d5 = BN_block3d(128, merge3d5)

    up3d6 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(conv3d5))
    ch, cw, cs = get_crop_shape(conv3d2, up3d6)
    crop_conv3d2 = Cropping3D(cropping=(ch, cw, cs))(conv3d2)
    merge3d6 = Concatenate()([crop_conv3d2, up3d6])
    conv3d6 = BN_block3d(64, merge3d6)

    up3d7 = Conv3D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=2)(conv3d6))
    ch, cw, cs = get_crop_shape(conv3d1, up3d7)
    crop_conv3d1 = Cropping3D(cropping=(ch, cw, cs))(conv3d1)
    merge3d7 = Concatenate()([crop_conv3d1, up3d7])
    conv3d7 = BN_block3d(32, merge3d7)

    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(conv3d7)
    conv8 = Lambda(squeeze)(x)
    conv9 = Conv2D(numClass, 1, activation='softmax')(conv8)

    model = Model(inputs=[inputs], outputs=[conv9])

    return model

#定义U-net
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

def Unet_convLSTM(numClass, NC, filter = [32,64,128]):

#   inputs = Input(shape=(NC, 96, 96, 1))
    inputs = Input(shape=(96, 96, NC))
    input3d = Permute((3, 1, 2))(inputs)
    input3d = Lambda(expand)(input3d)
    #print(input3d.get_shape())
    conv3d1 = BN_convLSTM(filter[0], input3d)
    #32*96*96*1


    conv3d1_mod = Lambda(squeeze)(conv3d1)
    conv3d1_mod = Permute((2, 3, 1))(conv3d1_mod)
    #96*96*32

    #print(conv3d1.get_shape())
#    print(conv3d1.get_shape())
#    conv3d1 = BN_convLSTM(32, inputs)
    pool3d1 = MaxPooling3D(pool_size=(1, 2, 2))(conv3d1)
    #32*48*48*1
    #print(pool3d1.get_shape())
    conv3d2 = BN_convLSTM(filter[1], pool3d1)
    #64*48*48*1
    conv3d2_mod = Lambda(squeeze)(conv3d2)
    conv3d2_mod = Permute(( 2, 3, 1))(conv3d2_mod)
    #48*48*64
    pool3d2 = MaxPooling3D(pool_size=(1, 2, 2))(conv3d2)
    #64*24*24*1
    conv3d3 = BN_convLSTM(filter[2], pool3d2)
    #128*24*24*1

    drop3d3 = Dropout( 0.3)(conv3d3)
    print(drop3d3.get_shape())

    drop3d3 = Lambda(squeeze)(drop3d3)
    #128*24*24
    drop3d3 = Permute(( 2, 3, 1,))(drop3d3)
    up3d4 = Conv2D(filter[1], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=2)(drop3d3))
    #48*48*64
    #ch, cw, cs = get_crop_shape(conv3d2, up3d4)
    #crop_conv3d2 = Cropping3D(cropping=(ch, cw, cs))(conv3d2)
    merge3d4 = Concatenate()([conv3d2_mod, up3d4])
    conv3d4 = BN_block(filter[1], merge3d4)
    #48*48*64
    up3d5 = Conv2D(filter[0], 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=2)(conv3d4))
    #96*96*32
    #ch, cw, cs = get_crop_shape(conv3d1, up3d5)
    #crop_conv3d1 = Cropping3D(cropping=(ch, cw, cs))(conv3d1)
    merge3d5 = Concatenate()([conv3d1_mod, up3d5])
    conv3d5 = BN_block(filter[0], merge3d5)
    #96*96*32
    #x = Conv2D(1, 1, padding='same', kernel_initializer='he_normal')(conv3d5)
    conv6 = Conv2D(numClass, 1, activation='softmax')(conv3d5)
    #96*96*5
    model = Model(inputs=[inputs], outputs=[conv6])

    return model

#method = 'SVD'
# method = 'PCA'
method = 'NoDR'
NC = 5
#保留spectral channel的数量
#model_Name = "2DUnet"
#model_Name = "Shallow"
model_Name = "Lightweight"
#model_Name = "LSTMUnet"
# model_Name = "Original"
numClass=6
#model = Unet(numClass,NC)
#NC = 251#保留spectral channel的数量

#model = Unet_3D(numClass,NC)
#plot_model(model,show_shapes = True,to_file = 'model.png')
#model = Unet(numClass)
#plot_model(model,show_shapes = True,to_file = 'model.png')
