import numpy as np
import os
from keras.utils import plot_model
#from keras.utils.vis_utils import plot_model
from model import *
from Statistics import *
from dataProc import *
from Dimension_reduction import *
#from mixup_generator import MixupGenerator
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Overall_Peform import Display_final_Table
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"]="1"


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)



def get_model_name(k):
    return 'model_'+str(k)+'.h5'

lr =2e-4
# lr =1e-4
img_size = [96, 96]

VALIDATION_ACCURACY = []
VALIDAITON_LOSS = []

history_dict = []

save_dir = 'saved_models/'
fold_var = 1



kf = KFold(n_splits = 3)
#原先三折
x_train=np.load('x_train_3DUnet_new.npy')
y_train=np.load('y_train_3DUnet_new.npy')
x_test = np.load('x_val_3DUnet_new.npy')
y_test = np.load('y_val_3DUnet_new.npy')
#RGB Unet的数据:
# x_train = np.load('xT_RGB.npy')
# y_train = np.load('yT_RGB.npy')
# x_test = np.load('xV_RGB.npy')
# y_test = np.load('yV_RGB.npy')


x_all = np.concatenate((x_train,x_test),axis = 0)
y_all = np.concatenate((y_train,y_test),axis = 0)
#将所有数据融合


#x_all_DR = Perform_DR(method,NC,x_train_aug)



for train_index, val_index in kf.split(x_all,y_all):
    x_train = x_all[train_index]
    y_train = y_all[train_index]
    x_train_aug, y_train_aug = load_data_aug(x_train, y_train, aug=4)
    #原先aug  = 4
    x_test = x_all[val_index]
    y_test = y_all[val_index]

    # CREATE NEW MODEL
    #模型，RGB Unet，需要修改
    model = Unet_Lightweight(numClass,NC)

    model.compile(optimizer=Adam(lr=lr), loss=focal_tversky_loss, metrics=[tversky])

    checkpoint = ModelCheckpoint(save_dir + get_model_name(fold_var),
                                                    monitor='tversky', verbose=1,
                                                    save_best_only=True, mode='max')

    earlyStopping = EarlyStopping(patience = 6, monitor = 'val_loss',mode = 'min')
    callbacks_list = [checkpoint,earlyStopping]
    history = model.fit(x_train_aug, y_train_aug, batch_size=1, epochs=30, callbacks = callbacks_list,validation_split=0.2)
    #原先split 0.2
    history_dict.append(history.history)
    ##mixup generator
    # training_generator = MixupGenerator(x_train, y_train, batch_size=1, alpha=0.2)()
    # history = model.fit_generator(generator=training_generator,
    #                     steps_per_epoch=x_train.shape[0],
    #                     epochs=30,
    #                     callbacks=callbacks_list)


    Display_final_Table(x_test, y_test, model)

    tf.keras.backend.clear_session()

    fold_var += 1


# index = np.argmax(np.array(VALIDATION_ACCURACY))
# print("best model: " + str(index+1))