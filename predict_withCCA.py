import numpy as np
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from model import *
from Statistics import *
from dataProc import *
import tensorflow as tf
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cca_post import CCA_postprocessing
from Dimension_reduction import *
K.set_image_data_format('channels_last')
from skimage import data
from skimage.morphology import square
from skimage.filters import rank
opt= mpimg.imread('newMScrop/Label/2_74.jpg')
from MSToGray import load_MS

model = Unet_Lightweight(numClass,NC)
model.load_weights("saved_models/model_" + str(1) + ".h5")
x_test = np.load('x_val_3DUnet.npy')
y_true = np.load('y_val_3DUnet.npy')


print(np.shape(x_test))#(23, 96, 96, 5)
print(np.shape(y_true))#(23, 96, 96, 6)

pred_test=model.predict(np.expand_dims(x_test[6],axis=0),verbose=1)
pred_test_t=pred_test.argmax(axis=-1)

#do cca_postprocessing to all classes to clear the noise
pred_0 = np.zeros(np.shape(pred_test_t))
pred_0[pred_test_t == 0] = 1
pred_0 = CCA_postprocessing(np.uint8(np.squeeze(pred_0)))

pred_1 = np.zeros(np.shape(pred_test_t))
pred_1[pred_test_t== 1] = 1
pred_1 = CCA_postprocessing(np.uint8(np.squeeze(pred_1)))

pred_2 = np.zeros(np.shape(pred_test_t))
pred_2[pred_test_t == 2] = 1
pred_2 = CCA_postprocessing(np.uint8(np.squeeze(pred_2)))

pred_3 = np.zeros(np.shape(pred_test_t))
pred_3[pred_test_t == 3] = 1
pred_3 = CCA_postprocessing(np.uint8(np.squeeze(pred_3)))

pred_4 = np.zeros(np.shape(pred_test_t))
pred_4[pred_test_t == 4] = 1
pred_4 = CCA_postprocessing(np.uint8(np.squeeze(pred_4)))

pred_5 = np.zeros(np.shape(pred_test_t))
pred_5[pred_test_t == 5] = 1
pred_5 = CCA_postprocessing(np.uint8(np.squeeze(pred_5)))


plt.figure()
ax1=plt.subplot(3,6,1)
plt.imshow(mask_to_rgba(pred_0,"gray"))
plt.xticks([])
plt.yticks([])
ax1.set_title("pre-background",fontsize=8)

ax2=plt.subplot(3,6,2)
plt.imshow(mask_to_rgba(pred_1,"royalblue"))
plt.xticks([])
plt.yticks([])
ax2.set_title("pre-1. layer",fontsize=8)

ax3=plt.subplot(3,6,3)
plt.imshow(mask_to_rgba(pred_2,"grassgreen"))
plt.xticks([])
plt.yticks([])
ax3.set_title("pre-2. layer",fontsize=8)

ax4=plt.subplot(3,6,4)
plt.imshow(mask_to_rgba(pred_3,"darkblue"))
plt.xticks([])
plt.yticks([])
ax4.set_title("pre-3. layer",fontsize=8)

ax5=plt.subplot(3,6,5)
plt.imshow(mask_to_rgba(pred_4,"gold"))
plt.xticks([])
plt.yticks([])
ax5.set_title("pre-multi. layer",fontsize=8)

ax6=plt.subplot(3,6,6)
plt.imshow(mask_to_rgba(pred_5,"orange"))
plt.xticks([])
plt.yticks([])
ax6.set_title("pre-bulk. layer",fontsize=8)

ax7 = plt.subplot(3,6,7)
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,0]),"gray"))
plt.xticks([])
plt.yticks([])
ax7.set_title("true-background",fontsize=8)

ax8 = plt.subplot(3,6,8)
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,1]),"royalblue"))
plt.xticks([])
plt.yticks([])
ax8.set_title("true-1. layer",fontsize=8)

ax9 = plt.subplot(3,6,9)
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,2]),"grassgreen"))
plt.xticks([])
plt.yticks([])
ax9.set_title("true-2. layer",fontsize=8)

ax10 = plt.subplot(3,6,10)
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,3]),"darkblue"))
plt.xticks([])
plt.yticks([])
ax10.set_title("true-3. layer",fontsize=8)

ax11 = plt.subplot(3,6,11)
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,4]),"gold"))
plt.xticks([])
plt.yticks([])
ax11.set_title("true-multi. layer",fontsize=8)

ax12 = plt.subplot(3,6,12)
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,5]),"orange"))
plt.xticks([])
plt.yticks([])
ax12.set_title("true-bulk. layer",fontsize=8)

ax13 = plt.subplot(3,6,13)
plt.imshow(opt)
plt.xticks([])
plt.yticks([])
ax13.set_title("Optical image",fontsize=8)

ax14 = plt.subplot(3,6,14)
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,0]),"gray"))
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,1]),"royalblue"))
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,2]),"grassgreen"))
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,3]),"darkblue"))
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,4]),"gold"))
plt.imshow(mask_to_rgba(np.squeeze(y_true[6,:,:,5]),"orange"))
plt.xticks([])
plt.yticks([])
ax14.set_title("true-mask",fontsize=8)



