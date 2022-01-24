import numpy as np
import os
from keras.utils import plot_model
from model import *
from Statistics import *
from dataProc import *
from sklearn.model_selection import KFold
import tensorflow as tf
from Overall_Peform import Display_final_Table
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

lr =2e-4
img_size = [96, 96]

history_dict = []

save_dir = 'saved_models/'
fold_var = 1

kf = KFold(n_splits = 3)

x_train=np.load('x_train_3DUnet_new.npy')
y_train=np.load('y_train_3DUnet_new.npy')
x_test = np.load('x_val_3DUnet_new.npy')
y_test = np.load('y_val_3DUnet_new.npy')

x_all = np.concatenate((x_train,x_test),axis = 0)
y_all = np.concatenate((y_train,y_test),axis = 0)

for train_index, val_index in kf.split(x_all,y_all):
    x_train = x_all[train_index]
    y_train = y_all[train_index]
    x_train_aug, y_train_aug = load_data_aug(x_train, y_train, aug=4)
    x_test = x_all[val_index]
    y_test = y_all[val_index]

    # CREATE NEW MODEL
    model = Unet_Lightweight(numClass,NC)
    model.compile(optimizer=Adam(lr=lr), loss=focal_tversky_loss, metrics=[tversky])

    checkpoint = ModelCheckpoint(save_dir + get_model_name(fold_var),
                                                    monitor='tversky', verbose=1,
                                                    save_best_only=True, mode='max')

    earlyStopping = EarlyStopping(patience = 6, monitor = 'val_loss',mode = 'min')
    callbacks_list = [checkpoint,earlyStopping]
    history = model.fit(x_train_aug, y_train_aug, batch_size=1, epochs=30, callbacks = callbacks_list,validation_split=0.2)
    history_dict.append(history.history)

    Display_final_Table(x_test, y_test, model)

    tf.keras.backend.clear_session()
    fold_var += 1
