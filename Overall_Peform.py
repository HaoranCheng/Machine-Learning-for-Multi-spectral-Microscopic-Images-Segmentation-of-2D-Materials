from __future__ import print_function
from __future__ import division
import SimpleITK as sitk
import prettytable
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from model import *
from dataProc import *

K.set_image_data_format('channels_last')
from evaluation import getDSC, getHausdorff, getVS
from prettytable import PrettyTable
from cca_post import CCA_postprocessing
from load_rgb_test import load_RGB_all
from MSToGray import load_MS
import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred):
    y_true = np.expand_dims(y_true, axis=0)
    true_class_0=y_true[0,:,:,0].astype(int)
    true_class_1=y_true[0,:,:,1].astype(int)
    true_class_2=y_true[0,:,:,2].astype(int)
    true_class_3=y_true[0,:,:,3].astype(int)
    true_class_4 =y_true[0,:,:,4].astype(int)
    true_class_5 = y_true[0, :, :, 5].astype(int)

    pre_class_0=y_pred[0,:,:,0].astype(int)
    pre_class_1 = y_pred[0, :, :, 1].astype(int)
    pre_class_2 = y_pred[0, :, :, 2].astype(int)
    pre_class_3 = y_pred[0, :, :, 3].astype(int)
    pre_class_4 = y_pred[0, :, :, 4].astype(int)
    pre_class_5 = y_pred[0, :, :, 5].astype(int)
    confusion_Sample = np.zeros((6,6))
    confusion_Sample[0]=[((true_class_0 * pre_class_0) == 1).sum(), ((true_class_0 * pre_class_1) == 1).sum(),
         ((true_class_0 * pre_class_2) == 1).sum(), ((true_class_0 * pre_class_3) == 1).sum(),((true_class_0 * pre_class_4) == 1).sum(),
         ((true_class_0 * pre_class_5) == 1).sum()]



    confusion_Sample[1]=[((true_class_1 * pre_class_0) == 1).sum(), ((true_class_1 * pre_class_1) == 1).sum(),
    ((true_class_1 * pre_class_2) == 1).sum(), ((true_class_1 * pre_class_3) == 1).sum(),((true_class_1 * pre_class_4) == 1).sum(),
         ((true_class_1 * pre_class_5) == 1).sum()]

    confusion_Sample[2]=[((true_class_2 * pre_class_0) == 1).sum(), ((true_class_2 * pre_class_1) == 1).sum(),
         ((true_class_2 * pre_class_2) == 1).sum(), ((true_class_2 * pre_class_3) == 1).sum(),((true_class_2 * pre_class_4) == 1).sum(),
         ((true_class_2 * pre_class_5) == 1).sum()]

    confusion_Sample[3]=[((true_class_3 * pre_class_0) == 1).sum(), ((true_class_3 * pre_class_1) == 1).sum(),
         ((true_class_3 * pre_class_2) == 1).sum(), ((true_class_3 * pre_class_3) == 1).sum(),((true_class_3 * pre_class_4) == 1).sum(),
         ((true_class_3 * pre_class_5) == 1).sum()]

    confusion_Sample[4]=[((true_class_4 * pre_class_0) == 1).sum(), ((true_class_4 * pre_class_1) == 1).sum(),
          ((true_class_4 * pre_class_2) == 1).sum(), ((true_class_4 * pre_class_3) == 1).sum(),((true_class_4 * pre_class_4) == 1).sum(),
         ((true_class_4 * pre_class_5) == 1).sum()]

    confusion_Sample[5]=[((true_class_5 * pre_class_0) == 1).sum(), ((true_class_5 * pre_class_1) == 1).sum(),
          ((true_class_5 * pre_class_2) == 1).sum(), ((true_class_5 * pre_class_3) == 1).sum(),((true_class_5 * pre_class_4) == 1).sum(),
         ((true_class_5 * pre_class_5) == 1).sum()]
    return confusion_Sample

def get_eval_metrics(true_mask, pred_mask):
    true_mask_sitk = sitk.GetImageFromArray(true_mask)
    pred_mask_sitk = sitk.GetImageFromArray(pred_mask)
    dsc = getDSC(true_mask_sitk, pred_mask_sitk)
    h95 = getHausdorff(true_mask_sitk, pred_mask_sitk)

    result = {}
    result['dsc'] = dsc
    #print(dsc)
    result['h95'] = h95
    #result['vs'] = vs

    return result 

def get_PredOnehot(x_testSample,y_trueSample,model):
    x_testSample = np.expand_dims(x_testSample,axis = 0)
    y_trueSample = np.expand_dims(y_trueSample,axis = 0)
    pred_test = model.predict(x_testSample)
    #1*96*96*5
    true_mask = y_trueSample.argmax(axis=-1)
    #1*96*96
    pred_test_t = pred_test.argmax(axis=-1)
    # cca processing 
    pred_0 = np.zeros(np.shape(pred_test_t))
    pred_0[pred_test_t == 0] = 1
    pred_0 = CCA_postprocessing(np.uint8(np.squeeze(pred_0)))
    #96*96
    pred_1 = np.zeros(np.shape(pred_test_t))
    pred_1[pred_test_t == 1] = 1
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

    pred_onehot = np.zeros((1, 96, 96, 6), dtype=np.int8)
    pred_onehot[:, :, :, 0] = pred_0
    pred_onehot[:, :, :, 1] = pred_1
    pred_onehot[:, :, :, 2] = pred_2
    pred_onehot[:, :, :, 3] = pred_3
    pred_onehot[:, :, :, 4] = pred_4
    pred_onehot[:, :, :, 5] = pred_5

    #  take the cca result of onehot,and merge them in to one channel segmentation result
    pred_mask_afterCCA = pred_onehot.argmax(axis=-1)
    #1*96*96
    result = get_eval_metrics(true_mask, pred_mask_afterCCA)
    return result,pred_onehot

def SaveAsTxt(path,table,method,model_Name,NC):
    f = open(path, "a")
    f.write(str(model_Name)+'_'+method+'_'+str(NC)+'\n')
    f.write(str(table) + '\n')
    f.close()

def construct_weights(y_true,Dice_weights):
    y_true = np.expand_dims(y_true, axis=0)
    true_class_0=y_true[0,:,:,0].astype(int).sum()
    true_class_1=y_true[0,:,:,1].astype(int).sum()
    true_class_2=y_true[0,:,:,2].astype(int).sum()
    true_class_3=y_true[0,:,:,3].astype(int).sum()
    true_class_4 =y_true[0,:,:,4].astype(int).sum()
    true_class_5 = y_true[0, :, :, 5].astype(int).sum()
    Dice_weights.append([true_class_0,true_class_1,true_class_2,true_class_3,true_class_4,true_class_5])

def Display_final_Table(x_test_DR,y_true,model):
    #performance
    T_H95 = PrettyTable(["H95", "Background", "1.Layer", "2.Layer", "3.Layer","MultiLayer","Bulk"])
    H95_mat = np.zeros((len(y_true),6))
    T_DSC = PrettyTable(["DSC", "Background", "1.Layer", "2.Layer", "3.Layer","MultiLayer","Bulk"])
    DSC_mat = np.zeros((len(y_true),6))
    T_Confusion = PrettyTable(["confusion", "Background", "1.Layer", "2.Layer", "3.Layer","MultiLayer","Bulk"])
    Confusion_mat = np.zeros((6,6))
    weights = []

    for i in range(0,len(y_true)):
        construct_weights(y_true[i],weights)
        result, pred_onehot =get_PredOnehot(x_test_DR[i],y_true[i],model)
        #get dsc and h95
        row_H = list(result['h95'].values())
        row_D = list(result['dsc'].values())
        H95_mat[i]=row_H
        DSC_mat[i]=row_D
        row_H.insert(0,"Sample"+str(i))
        row_D.insert(0,"Sample"+str(i))
        T_H95.add_row(row_H)
        T_DSC.add_row(row_D)
        #get confusion matrix of each sample
        Confusion_mat += confusion_matrix(y_true[i], pred_onehot)
    #creat Confusion Table
    row_C = list(Confusion_mat[0])
    row_C.insert(0, "Background")
    T_Confusion.add_row(row_C)
    row_C = list(Confusion_mat[1])
    row_C.insert(0, "1.Layer")
    T_Confusion.add_row(row_C)
    row_C = list(Confusion_mat[2])
    row_C.insert(0, "2.Layer")
    T_Confusion.add_row(row_C)
    row_C = list(Confusion_mat[3])
    row_C.insert(0, "3.Layer")
    T_Confusion.add_row(row_C)
    row_C = list(Confusion_mat[4])
    row_C.insert(0, "MultiLayer")
    T_Confusion.add_row(row_C)
    row_C = list(Confusion_mat[5])
    row_C.insert(0, "Bulk")
    T_Confusion.add_row(row_C)

    nan_mask = np.isnan(H95_mat)
    H95_mat[nan_mask]= 0
    weights = np.asarray(weights)
    weights = weights/np.sum(weights,axis=0)

    row_Haverage = list(np.sum(weights*H95_mat,axis=0))
    row_Haverage.insert(0, "AverageScore:")
    T_H95.add_row(row_Haverage)

    nan_mask = np.isnan(DSC_mat)
    DSC_mat[nan_mask]= 0

    row_Daverage = list(np.sum(weights*DSC_mat,axis=0))
    row_Daverage.insert(0, "AverageScore:")

    T_DSC.add_row(row_Daverage)

    T_H95.set_style(prettytable.PLAIN_COLUMNS)
    T_DSC.set_style(prettytable.PLAIN_COLUMNS)
    T_Confusion.set_style(prettytable.PLAIN_COLUMNS)
    print(T_H95)
    print(T_DSC)
    print(T_Confusion)

    SaveAsTxt("Result/T_H95.txt", T_H95, method, model_Name, NC)
    SaveAsTxt("Result/T_DSC.txt", T_DSC, method, model_Name, NC)
    SaveAsTxt("Result/T_Confusion.txt", T_Confusion, method, model_Name, NC)

def main():
    model = Unet_Lightweight(numClass, NC)
    model.load_weights(str(NC) +'_'+ model_Name + '_' + method + '.h5')
    model.load_weights("saved_models/model_" + str(2) + ".h5")
    x_test = np.load('x_val_3DUnet.npy')
    y_true = np.load('y_val_3DUnet.npy')
    Display_final_Table(x_test,y_true,model)
if __name__ == "__main__":
    main()



