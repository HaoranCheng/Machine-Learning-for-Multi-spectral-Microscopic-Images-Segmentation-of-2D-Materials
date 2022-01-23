from calendar import c
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5.Qt import QImage, QPixmap, QPalette, QColor
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QPalette
#from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtCore import Qt
import tkinter as tk
#from tkinter import filedialog
from magicgui.backends._qtpy import show_file_dialog
from tkinter import filedialog
#file_path = show_file_dialog()
from PyQt5.QtGui import qRgb
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt1
import PIL
from matplotlib import ticker
from PIL import ImageQt
from magicgui.backends._qtpy import show_file_dialog
from PIL import Image
import os
import shutil
from tkinter.filedialog import test
import numpy as np
from cv2 import imread
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from model import *

import tensorflow as tf
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

K.set_image_data_format('channels_last')
from skimage import data
from skimage.morphology import square
from skimage.filters import rank
#opt= mpimg.imread('newMScrop/Label/2_74.jpg')


from MSToGray import load_MS
from cca_post import CCA_postprocessing
from Statistics import *
from dataProc import *
from Dimension_reduction import *

class fileDialogdemo(QWidget):
    global inputFilePath, result, outputFilePath, histogram, histoFilePath,IMAGE_SAVE_PATH
    sample = 'sample'
    substrates = 'substrates'
    objective = 0
    load = False

    def __init__(self,parent=None):
        super(fileDialogdemo, self).__init__(parent)
 
    #垂直布局
        layout=QGridLayout()#QHBoxLayout()
        
   #创建按钮，绑定自定义的槽函数，添加到布局中
        #self.btn=QPushButton("加载图片")
        #self.btn.clicked.connect(self.getimage)
        #layout.addWidget(self.btn)
        layout
        
        self.OpenPicture = QPushButton("Load Image")
        self.OpenPicture.setGeometry(QtCore.QRect(0, 0, 11, 16))
        self.OpenPicture.setObjectName("OpenPicture")
        self.OpenPicture.clicked.connect(self.getimage)
        layout.addWidget(self.OpenPicture, 0 , 0)
        
       
        self.MS_Picture = QPushButton("Load MS_Image")
        self.MS_Picture.setGeometry(QtCore.QRect(0, 0, 11, 16))
        self.MS_Picture.setObjectName("MS_Picture")
        self.MS_Picture.clicked.connect(self.get_ms_image)
        layout.addWidget(self.MS_Picture, 0, 1)
        
        self.PredictPicture =QPushButton('Predict')
        self.PredictPicture.setGeometry(QtCore.QRect(0, 0, 11, 16))
        self.PredictPicture.setObjectName("PredictPicture")
        self.PredictPicture.clicked.connect(self.predictPicture)
        layout.addWidget(self.PredictPicture, 0 , 2,)
         
        '''self.Open_pre_Picture = QPushButton("Load_pred_Image")
        self.Open_pre_Picture.setGeometry(QtCore.QRect(0, 0, 11, 16))
        self.Open_pre_Picture.setObjectName("Open_pre_Picture")
        self.Open_pre_Picture.clicked.connect(self.get_pre_image)
        layout.addWidget(self.Open_pre_Picture, 0 , 3)
        self.Open_pre_Picture.setEnabled(False)'''
        
        
      
    #创建标签，添加到布局中
        self.le=QLabel('')
        self.le.resize(100, 100)
        layout.addWidget(self.le,1,0)
        self.le.setFixedSize(QSize(256, 192))
        
        self.le_=QLabel('RGB')
        self.le_.resize(100, 100)
        layout.addWidget(self.le_,2,0)
        self.le_.setFixedSize(QSize(50, 20))
        self.le_.setVisible(False)
        
        self.nm520=QLabel('')
        self.nm520.resize(100, 100)
        layout.addWidget(self.nm520,3,0)
        self.nm520.setFixedSize(QSize(256, 192))
        
        self.nm520_=QLabel('520nm')
        self.nm520_.resize(100, 100)
        layout.addWidget(self.nm520_,4,0)
        self.nm520_.setFixedSize(QSize(50, 20))
        self.nm520_.setVisible(False)
        
        self.nm540=QLabel('')
        self.nm540.resize(100, 100)
        layout.addWidget(self.nm540,5,0)
        self.nm540.setFixedSize(QSize(256, 192))
        
        self.nm540_=QLabel('540nm')
        self.nm540_.resize(100, 100)
        layout.addWidget(self.nm540_,6,0)
        self.nm540_.setFixedSize(QSize(50, 20))
        self.nm540_.setVisible(False)
        
        self.nm560=QLabel('')
        self.nm560.resize(100, 100)
        layout.addWidget(self.nm560,1,1)
        self.nm560.setFixedSize(QSize(256, 192))
        
        self.nm560_=QLabel('560nm')
        self.nm560_.resize(100, 100)
        layout.addWidget(self.nm560_,2,1)
        self.nm560_.setFixedSize(QSize(50, 20))
        self.nm560_.setVisible(False)
        
        self.nm580=QLabel('')
        self.nm580.resize(100, 100)
        layout.addWidget(self.nm580,3,1)
        self.nm580.setFixedSize(QSize(256, 192))

        self.nm580_=QLabel('580nm')
        self.nm580_.resize(100, 100)
        layout.addWidget(self.nm580_,4,1)
        self.nm580_.setFixedSize(QSize(50, 20))
        self.nm580_.setVisible(False)
        
        self.nm600=QLabel('')
        self.nm600.resize(100, 100)
        layout.addWidget(self.nm600,5,1)
        self.nm600.setFixedSize(QSize(256, 192))
        
        self.nm600_=QLabel('600nm')
        self.nm600_.resize(100, 100)
        layout.addWidget(self.nm600_,6,1)
        self.nm600_.setFixedSize(QSize(50, 20))
        #self.nm_600.clicked.connect(self.get_ms_image)
        self.nm600_.setVisible(False)
        
        self.predicted=QLabel('')
        self.predicted.resize(100, 100)
        layout.addWidget(self.predicted,1,2,4,6)
        self.predicted.setFixedSize(QSize(512, 384))
        
        self.predicted_=QLabel('Prediction')
        self.predicted_.resize(100, 100)
        layout.addWidget(self.predicted_,4,2)
        self.predicted_.setFixedSize(QSize(85, 20))
        self.predicted_.setVisible(False)
        class Color(QWidget):
            def __init__(self, color, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.setAutoFillBackground(True)

                palette = self.palette()
                palette.setColor(QPalette.Window, QColor(color))
                self.setPalette(palette)
        #self.color=
        colors = ["gray","dodgerblue","#7FFFD4","darkblue","gold","orange"]
        #for i, color in zip(range(3,9),colors):
            #layout.addWidget(Color(color), i,2,6,7)
        self.color0=Color(colors[0])
        self.color1=Color(colors[1])
        self.color2=Color(colors[2])
        self.color3=Color(colors[3])
        self.color4=Color(colors[4])
        self.color5=Color(colors[5])
        
        self.color0.setVisible(False)
        self.color1.setVisible(False)
        self.color2.setVisible(False)
        self.color3.setVisible(False)
        self.color4.setVisible(False)
        self.color5.setVisible(False)
        
        #色块的大小
        self.color0.setFixedSize(50,50)
        self.color1.setFixedSize(50,50)
        self.color2.setFixedSize(50,50)
        self.color3.setFixedSize(50,50)
        self.color4.setFixedSize(50,50)
        self.color5.setFixedSize(50,50)
        #色块的位置
        layout.addWidget(self.color0, 5,2,6,3)
        layout.addWidget(self.color1, 5,3,6,4)
        layout.addWidget(self.color2, 5,4,6,5)
        layout.addWidget(self.color3, 5,5,6,6)
        layout.addWidget(self.color4, 5,6,6,7)
        layout.addWidget(self.color5, 5,7,6,8)
        #标签的大小和位置
        self.color0_=QLabel('Background')
        layout.addWidget(self.color0_,5,2)
        self.color0_.setFixedSize(QSize(85, 20))
        #self.color0_.setVisible(False)
        
        self.color1_=QLabel('Monolayer')
        layout.addWidget(self.color1_,5,3)
        self.color1_.setFixedSize(QSize(85, 20))
        #self.color1_.setVisible(False)
        
        self.color2_=QLabel('Bilayer')
        layout.addWidget(self.color2_,5,4)
        self.color2_.setFixedSize(QSize(85, 20))
        #self.color2_.setVisible(False)
        
        self.color3_=QLabel('Trilayer')
        layout.addWidget(self.color3_,5,5)
        self.color3_.setFixedSize(QSize(85, 20))
        #self.color3_.setVisible(False)
        
        self.color4_=QLabel('Multilayer')
        layout.addWidget(self.color4_,5,6)
        self.color4_.setFixedSize(QSize(85, 20))
        #self.color4_.setVisible(False)

        self.color5_=QLabel('Bulk')
        layout.addWidget(self.color5_,5,7)
        self.color5_.setFixedSize(QSize(85, 20))
        #self.color5_.setVisible(False)
        
        
        
        self.color0_.setVisible(False)
        self.color1_.setVisible(False)
        self.color2_.setVisible(False)
        self.color3_.setVisible(False)
        self.color4_.setVisible(False)
        self.color5_.setVisible(False)
        
        #widget = QWidget()
        #widget.setLayout(layout)
        
      
        

        '''
        self.Input = QPushButton('Input')
        #self.Input.setGeometry(QtCore.QRect(40, 50, 360, 271))
        self.Input.setText("")
        self.Input.setObjectName("Input")
        input = QPalette()
        input.setColor(QPalette.Background, Qt.white)
        self.Input.setAutoFillBackground(True)
        self.Input.setPalette(input)
        
        
        self.Output = QPushButton('Output')
        self.Output.setGeometry(QtCore.QRect(40, 390, 360, 271))
        self.Output.setText("")
        self.Output.setObjectName("Output")
        output = QPalette()
        output.setColor(QPalette.Background, Qt.white)
        self.Output.setAutoFillBackground(True)
        self.Output.setPalette(output)
        '''
    #设置主窗口的布局及标题
        self.setLayout(layout)
        self.setWindowTitle('Multispectral Image Segmentation GUI with 3D-Unet')
 
    def getimage(self):
    #从C盘打开文件格式（*.jpg *.gif *.png *.jpeg）文件，返回路径
        image_file,_=QFileDialog.getOpenFileName(self,'Open file','C:\\','Image files (*.jpg *.gif *.png *.jpeg)')
    #设置标签的图片
        #print(type(image_file),image_file)
        #filename, filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "Image files (*.jpg *.gif *.png *.jpeg)")  #设置文件扩展名过滤,注意用双分号间隔
    #设置标签的图片
        imm = Image.open(image_file)
        outfile = image_file.replace('.jpg','_resize.jpg')
        out = imm.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile)
        #out file 是缩小后的
        self.le.setPixmap(QPixmap(outfile))
        self.le.resize(100, 100)
        self.inputFilePath=image_file
        self.le_.setVisible(True)
    def get_ms_image(self):
        
        ms_resize_folder=self.inputFilePath.replace('.jpg','_resize')
        ms_foler=self.inputFilePath.replace('.jpg','')+'/'
        try:
            os.mkdir(ms_resize_folder)# 创建额外的region_resize文件夹
        except:
            pass
        
        ms_file_list=os.listdir(ms_foler)#ms的图片list ".jpg",".jpg"
        if '.DS_Store' in ms_file_list:
            ms_file_list.remove('.DS_Store')
        ms_file_list.sort(key=lambda x:(int(x.split('_')[-1].replace('nm.jpg',''))))
        ms_file_list.remove(ms_file_list[0])
        #print(ms_file_list)
        nm_list=["nm520","nm540","nm560","nm580","nm600"]
        
        imm_520 = Image.open(os.path.join(ms_foler,ms_file_list[0]))
        outfile_520 = os.path.join(ms_resize_folder,ms_file_list[0].replace('.jpg','_resize.jpg'))
        out = imm_520.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_520)
        #out file 是缩小后的
        self.nm520.setPixmap(QPixmap(outfile_520))
        self.nm520.resize(100, 100)
        
        imm_540 = Image.open(os.path.join(ms_foler,ms_file_list[1]))
        outfile_540 = os.path.join(ms_resize_folder,ms_file_list[1].replace('.jpg','_resize.jpg'))
        out = imm_540.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_540)
        #out file 是缩小后的
        self.nm540.setPixmap(QPixmap(outfile_540))
        self.nm540.resize(100, 100)
        
        imm_560 = Image.open(os.path.join(ms_foler,ms_file_list[2]))
        outfile_560 = os.path.join(ms_resize_folder,ms_file_list[2].replace('.jpg','_resize.jpg'))
        out = imm_560.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_560)
        #out file 是缩小后的
        self.nm560.setPixmap(QPixmap(outfile_560))
        self.nm560.resize(100, 100)
        
        imm_580 = Image.open(os.path.join(ms_foler,ms_file_list[3]))
        outfile_580 = os.path.join(ms_resize_folder,ms_file_list[3].replace('.jpg','_resize.jpg'))
        out = imm_580.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_580)
        #out file 是缩小后的
        self.nm580.setPixmap(QPixmap(outfile_580))
        self.nm580.resize(100, 100)
        
        imm_600 = Image.open(os.path.join(ms_foler,ms_file_list[4]))
        outfile_600 = os.path.join(ms_resize_folder,ms_file_list[4].replace('.jpg','_resize.jpg'))
        out = imm_600.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_600)
        #out file 是缩小后的
        self.nm600.setPixmap(QPixmap(outfile_600))
        self.nm600.resize(100, 100)
        
        self.nm520_.setVisible(True)
        self.nm540_.setVisible(True)
        self.nm560_.setVisible(True)
        self.nm580_.setVisible(True)
        self.nm600_.setVisible(True)
        
        
        
        
        try:
            shutil.rmtree(ms_resize_folder)
        except:
            pass
        
    
    def predictPicture(self):
        #print(self.inputFilePath)
        #裁剪并且，每张小图设置单独文件夹
        def cut_image(path,newpath):
            #print('Original Image:',os.listdir(path))
            image_list = os.listdir(path)
            if '.DS_Store' in image_list:
                image_list.remove('.DS_Store')
    #image_list.sort(key=lambda x:(int(x.split('_')[0]),int(x.split('_')[-1])))
            for ms in image_list:
                img = Image.open(path+f"{ms}")
        #print(ms,img.size)
                try:
                    os.mkdir(newpath)
                except:
                    pass
                    #print('already exist')
                for high in range(8):
                    for length in range(10):
                        cropped = img.crop((length*96, high*96,(length+1)*96 , (high+1)*96))  # (left, upper, right, lower)
                        cropped.save(newpath+f"{ms.replace('.jpg','')}_{high+1}_{length+1}.jpg")
                    cropped_extra = img.crop((928, high*96,1024 , (high+1)*96))  # (left, upper, right, lower)
                    cropped_extra.save(newpath+f"{ms.replace('.jpg','')}_{high+1}_11.jpg")

        def set_seperate_folder(path,newpath):
    #path= '/Users/haoran/Desktop/newregions/'
    #newpath= '/Users/haoran/Desktop/data/'
            train_list=os.listdir(path)
#print(train_list)
            try:
                os.mkdir(newpath)
            except:
                pass#print('newpath exists')
            folder_list= os.listdir(newpath)
    #print(folder_list)
            if '.DS_Store' in train_list:
                train_list.remove('.DS_Store')
            for file in train_list:
                folder_name =file.split('_')[-2]+'_'+ file.split('_')[-1].replace('.jpg','')
        #print(folder_name)
                if  folder_name in folder_list:
                    shutil.copyfile(path+file,newpath+folder_name+'/'+file)
        #shutil.move(path+file,newpath+folder_name)
                elif folder_name not in folder_list:
                    try:
                        os.mkdir(newpath+folder_name)
                    except:
                        pass#print('exists')
            #print(folder_name)
                    shutil.copyfile(path+file,newpath+folder_name+'/'+file)
                    #print('New folder created')
            
        def Concat_RGB_MS(MS_data,list):
            data_path = 'MultiSpectral/RGBResize'
            temp = np.zeros((len(list),96,96,7))
            for i in range(len(list)):
                img = Image.open(os.path.join(data_path, list[i] + '.png'))
                img = np.asarray(img.convert('RGB'))
                RGB = np.asarray(img)
        #print("RGB max"+str(np.max(RGB)),"RGB min"+str(np.min(RGB)))
                temp[i] = np.concatenate((RGB, np.squeeze(MS_data[i])), axis=-1)
            return temp

        def MSInterpolation(img):
            #print("Original:"+str(np.shape(img)))
            weight1 = [0.5,0,0.5,0]
            weight2 = [0.5,0.5,0,0]
            weight3 = [0,0.5,0.5,0]
            weight4 = [0.5,0,0,0.5]
            temp = np.zeros((len(img),96,96,4))
            for i in range(0,4):
                temp[:,:,:,i]=weight1[i]*img[:,:,:,0]+weight2[i]*img[:,:,:,1]+weight3[i]*img[:,:,:,2]+weight4[i]*img[:,:,:,3]
            img = np.concatenate((img,temp),axis=-1)
            #print(np.shape(img))
            return img

        def Layer_extraction(data, mask,number):
            new_mask = mask[:,:,number].astype(np.int)
    #new_mask 96*96
    #data 96*96*251
    #layer 96*96*251
    #exists 96*96
            layer = data*(np.expand_dims(new_mask,axis =2))
            exists = (layer[:,:,0] != 0)
            if exists.sum() == 0:
                return None
            result = np.sum(np.sum(layer, axis=0), axis=0) / exists.sum()
    #result 1*251
            return result


        def load_MS(data_path):
    ## one-hot conversion
            def convert_to_onehot(label, numClass):

                one_hot = np.zeros((1, label.shape[0], label.shape[1], numClass), dtype=np.float32)
                for i in range(numClass):
                    one_hot[0, :, :, i][label == i+1] = 1
                return one_hot

    ## paramters of the image size
    #     IMG_WIDTH = 96
    #     IMG_HEIGHT = 96
    #     IMG_CHANNELS = 254
            NUM_class = 6
    #data_path = 'newMScrop/cheng/'

            X_val = []  # validation data
 
    
            temp_path = os.path.join(data_path)
    #print(temp_path)
    #print(temp_path)
            test_list = sorted(os.listdir(temp_path))
            if ".DS_Store" in test_list:
                test_list.remove(".DS_Store")
    #train_list = sorted(os.listdir(temp_path))
            test_list.sort(key=lambda x:(int(x.split('_')[0]),int(x.split('_')[-1].replace('.jpg',''))))
    #print(test_list)
    
            if len(test_list)!=0:
                for nn in test_list:
                    temp_path1 = temp_path + '/' + nn
                    img_list = sorted(os.listdir(temp_path1))
                    single_train = []
                    for file in img_list:
                 #print(file)
                        img = imread(temp_path1+'/'+file)
                        img = np.dot(img, [0.299, 0.587, 0.114])
                 #img = Channel_normalization(np.array(img))
                        single_train.append(img)
                    single_train = np.transpose(single_train,(1,2,0))
                    X_val.append(single_train)
             #print(np.shape(X_train))
             #label = np.asarray(Image.open(os.path.join(data_path, 'Label', nn+'_Simple Segmentation.png')))
             #print(np.max(label))
             #print(np.shape(label))
             #print(np.shape(label))
             #one_hot_label = np.squeeze(convert_to_onehot(label, NUM_class))
             #print(none_hot_label[:,:,3]))
             #y_val.append(one_hot_label)
        #删除第一channel，删掉那个较暗的波长
                X_val = np.asarray(X_val)
                X_val = X_val[:,:,:,1:6]
        #y_val = np.asarray(y_val)

        #light compensation
        #X_val[-2:] = X_val[-2:]*Bg_before/Bg_after
        # X_val[2:4] = X_val[2:4]*Back_ground3/Back_ground4

        #channel normalization
                max_channel = np.max(np.max(np.max(X_val,axis=0),axis=0),axis=0)
                min_channel = np.min(np.min(np.min(X_val,axis=0),axis=0),axis=0)
                for i in range(len(X_val)):
                    for j in range(X_val.shape[3]):
                        X_val[i, :, :, j] = (X_val[i, :, :, j] - min_channel[j]) / (max_channel[j] - min_channel[j]) * 255
        #channel Interpolation
        #X_val = MSInterpolation(X_val)
        #X_val = Concat_RGB_MS(X_val, val_list)
            return X_val
#设置路径*********************************************************    
        inputfile_foler=self.inputFilePath.replace('.jpg','')
        try:
            os.mkdir(inputfile_foler)
        except:
            pass#print('folder existed')
        ms_path=inputfile_foler+'/'#os.getcwd()+"/GUI_cheng/newregion1/"
        cutted_path=os.path.join(os.path.dirname(sys.argv[0]), 'allimages/')#os.getcwd()+"/GUI_cheng/newregion1/allimages/"#mkdir        
        seperated_path=os.path.join(os.path.dirname(sys.argv[0]), 'seperate/')#os.getcwd()+"/GUI_cheng/newregion1/seperate/"#mkdir
        #load_model_path=os.getcwd()+"/saved_models/chr/model_3lsame_gamma_aug_1.h5"
        load_model_path=os.path.join(os.path.dirname(sys.argv[0]), "saved_models/model_3.h5")
#*********************************************************      

#裁剪所有所有nm的大图片并放在一个文件夹           
#cut_image(os.getcwd()+"/GUI_cheng/newregion1",os.getcwd()+"/GUI_cheng/newregion1/allimages")       
#set_seperate_folder(os.getcwd()+"/GUI_cheng/newregion1/allimages/",os.getcwd()+"/GUI_cheng/newregion1/sepeate/")
        cut_image(ms_path,cutted_path)#seperate 的path是cutimgae的newpath
        set_seperate_folder(cutted_path,seperated_path)
        try:
            shutil.rmtree(cutted_path)
        except:
            pass
        X_val= load_MS(seperated_path)
        try:
            shutil.rmtree(seperated_path)
        except:
            pass
#load_MS的path是seperate的路径)
        #print("Val"+str(np.shape(X_val)))
        np.save('x_val_3DUnet_gui.npy', X_val)
#预测小图片并放在precicted_path
        model = Unet_Lightweight(numClass,NC)
        model_choice='_new_all_2_'#'_'#_new_  #_new_all_
        model.load_weights(load_model_path)
        x_test = np.load('x_val_3DUnet_gui.npy')

        x_test_DR = Perform_DR(method,NC,x_test)
        #print('npy file.shape:',x_test_DR.shape)
        try :
            os.mkdir(os.path.join(os.path.dirname(sys.argv[0]), 'pred_image'))
            
        except:
            pass#print('existed')
        predicted_path=os.path.join(os.path.dirname(sys.argv[0]), 'pred_image/')
        
        for imm in range(88):
#pred_test_patch=model.predict(x_test_patch,verbose=1)
            pred_test=model.predict(np.expand_dims(x_test_DR[imm],axis=0),verbose=1)

#pred_test_patch=median_f(pred_test_patch,3)


            pred_test_t=pred_test.argmax(axis=-1)

# 分别对每一个class的结果做cca_postprocessing,去除杂点
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

            fig=plt.figure()
            ax14 = fig.add_subplot(3,6,14)    
            plt.imshow(mask_to_rgba(pred_0,"gray"))
            plt.imshow(mask_to_rgba(pred_1,"royalblue"))
            plt.imshow(mask_to_rgba(pred_2,"grassgreen"))
            plt.imshow(mask_to_rgba(pred_3,"darkblue"))
            plt.imshow(mask_to_rgba(pred_4,"gold"))
            plt.imshow(mask_to_rgba(pred_5,"orange"))
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.savefig(predicted_path+'GUI_2.jpg')
            extent = ax14.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(predicted_path+f'{imm}.jpg', bbox_inches=extent)
    #plt.show()
            pic = Image.open(predicted_path+f'{imm}.jpg')
            pic = pic.resize((96, 96))
            pic.save(predicted_path+f'{imm}.jpg')
        for x in range(1,9):
    
            img = Image.open(predicted_path+f'{x*11-1}.jpg')
            cropped = img.crop((32, 0, 96, 96))  # (left, upper, right, lower)
            cropped.save(predicted_path+f'{x*11-1}.jpg')
#预测后的图片在predicted_path里面
#融合预测的图片从Images_PATH到IMAGE_SAVE_PATH
        IMAGES_PATH = predicted_path # 图片集地址  
        IMAGES_FORMAT = ['.jpg', '.JPG'] # 图片格式  
        IMAGE_SIZE = 96 # 每张小图片的大小  
        IMAGE_ROW = 8 # 图片间隔，也就是合并成一张图后，一共有几行  
        IMAGE_COLUMN = 11 # 图片间隔，也就是合并成一张图后，一共有几列  
        cccc=load_model_path.split('/')[-1].replace('.h5','')
        self.IMAGE_SAVE_PATH = os.path.join(os.path.dirname(sys.argv[0]), 'final.jpg') # 图片转换后的地址  
        

# 获取图片集地址下的所有图片名称  
        image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if  
        os.path.splitext(name)[1] == item]  
#image_names=os.listdir
        image_names = sorted(os.listdir(IMAGES_PATH))
        image_names.remove("GUI_2.jpg")
        if ".DS_Store" in image_names:
            image_names.remove(".DS_Store")
        image_names.sort(key=lambda x:(int(x.split('.')[0])))
        #print(image_names)
  
# 定义图像拼接函数  
        def image_compose():  
            to_image = Image.new('RGB', (1024, 768)) #创建一个新图  
# 循环遍历，把每张图片按顺序粘贴到对应位置上  
            for y in range(1, IMAGE_ROW + 1):  
                for x in range(1, IMAGE_COLUMN + 1):  
                    from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1])
           
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))  
            return to_image.save(self.IMAGE_SAVE_PATH) # 保存新图  
        image_compose() #调用函数  
        try:
            shutil.rmtree(IMAGE_SIZE)
        except:
            pass
        #self.Open_pre_Picture.setEnabled(True)
        predicted_file=self.IMAGE_SAVE_PATH
        #filename, filetype = QFileDialog.getOpenFileName(self,"选取文件","./", "Image files (*.jpg *.gif *.png *.jpeg)")  #设置文件扩展名过滤,注意用双分号间隔
    #设置标签的图片
        self.predicted.setPixmap(QPixmap(predicted_file))
        self.predicted.resize(100, 100)
        
        
        imm_pre = Image.open(predicted_file)
        outfile_pre = predicted_file.replace('.jpg','_resize.jpg')
        out_pre = imm_pre.resize((512,384),Image.ANTIALIAS) #resize image with high-quality
        out_pre.save(outfile_pre)
        #out file 是缩小后的
        self.predicted.setPixmap(QPixmap(outfile_pre))
        self.predicted.resize(100, 100)
        self.predicted_.setVisible(True)
        self.color0.setVisible(True)
        self.color1.setVisible(True)
        self.color2.setVisible(True)
        self.color3.setVisible(True)
        self.color4.setVisible(True)
        self.color5.setVisible(True)
        self.color0_.setVisible(True)
        self.color1_.setVisible(True)
        self.color2_.setVisible(True)
        self.color3_.setVisible(True)
        self.color4_.setVisible(True)
        self.color5_.setVisible(True)

if __name__ == '__main__':
  app=QApplication(sys.argv)
  ex=fileDialogdemo()
  ex.resize(680,550)
  ex.show()
  sys.exit(app.exec_())