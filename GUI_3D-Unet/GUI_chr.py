from calendar import c
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QPalette
from PyQt5.QtCore import Qt
import tkinter as tk
from magicgui.backends._qtpy import show_file_dialog
from tkinter import filedialog
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

from MSToGray import load_MS
from cca_post import CCA_postprocessing
from Statistics import *
from dataProc import *

class fileDialogdemo(QWidget):
    global inputFilePath, result, outputFilePath, histogram, histoFilePath,IMAGE_SAVE_PATH
    sample = 'sample'
    substrates = 'substrates'
    objective = 0
    load = False

    def __init__(self,parent=None):
        super(fileDialogdemo, self).__init__(parent)
        layout=QGridLayout()#QHBoxLayout()       
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
        colors = ["gray","dodgerblue","#7FFFD4","darkblue","gold","orange"]
  
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

        self.color0.setFixedSize(50,50)
        self.color1.setFixedSize(50,50)
        self.color2.setFixedSize(50,50)
        self.color3.setFixedSize(50,50)
        self.color4.setFixedSize(50,50)
        self.color5.setFixedSize(50,50)

        layout.addWidget(self.color0, 5,2,6,3)
        layout.addWidget(self.color1, 5,3,6,4)
        layout.addWidget(self.color2, 5,4,6,5)
        layout.addWidget(self.color3, 5,5,6,6)
        layout.addWidget(self.color4, 5,6,6,7)
        layout.addWidget(self.color5, 5,7,6,8)

        self.color0_=QLabel('Background')
        layout.addWidget(self.color0_,5,2)
        self.color0_.setFixedSize(QSize(85, 20))
        
        self.color1_=QLabel('Monolayer')
        layout.addWidget(self.color1_,5,3)
        self.color1_.setFixedSize(QSize(85, 20))
        
        self.color2_=QLabel('Bilayer')
        layout.addWidget(self.color2_,5,4)
        self.color2_.setFixedSize(QSize(85, 20))
        
        self.color3_=QLabel('Trilayer')
        layout.addWidget(self.color3_,5,5)
        self.color3_.setFixedSize(QSize(85, 20))
        
        self.color4_=QLabel('Multilayer')
        layout.addWidget(self.color4_,5,6)
        self.color4_.setFixedSize(QSize(85, 20))

        self.color5_=QLabel('Bulk')
        layout.addWidget(self.color5_,5,7)
        self.color5_.setFixedSize(QSize(85, 20))
        
        self.color0_.setVisible(False)
        self.color1_.setVisible(False)
        self.color2_.setVisible(False)
        self.color3_.setVisible(False)
        self.color4_.setVisible(False)
        self.color5_.setVisible(False)
        
        self.setLayout(layout)
        self.setWindowTitle('Multispectral Image Segmentation GUI with 3D-Unet')
 
    def getimage(self):
        image_file,_=QFileDialog.getOpenFileName(self,'Open file','C:\\','Image files (*.jpg *.gif *.png *.jpeg)')
        imm = Image.open(image_file)
        outfile = image_file.replace('.jpg','_resize.jpg')
        out = imm.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile)
        #out file is resized 
        self.le.setPixmap(QPixmap(outfile))
        self.le.resize(100, 100)
        self.inputFilePath=image_file
        self.le_.setVisible(True)
    def get_ms_image(self):
        
        ms_resize_folder=self.inputFilePath.replace('.jpg','_resize')
        ms_foler=self.inputFilePath.replace('.jpg','')+'/'
        try:
            os.mkdir(ms_resize_folder)
        except:
            pass
        
        ms_file_list=os.listdir(ms_foler)
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
        self.nm520.setPixmap(QPixmap(outfile_520))
        self.nm520.resize(100, 100)
        
        imm_540 = Image.open(os.path.join(ms_foler,ms_file_list[1]))
        outfile_540 = os.path.join(ms_resize_folder,ms_file_list[1].replace('.jpg','_resize.jpg'))
        out = imm_540.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_540)
        self.nm540.setPixmap(QPixmap(outfile_540))
        self.nm540.resize(100, 100)
        
        imm_560 = Image.open(os.path.join(ms_foler,ms_file_list[2]))
        outfile_560 = os.path.join(ms_resize_folder,ms_file_list[2].replace('.jpg','_resize.jpg'))
        out = imm_560.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_560)
        self.nm560.setPixmap(QPixmap(outfile_560))
        self.nm560.resize(100, 100)
        
        imm_580 = Image.open(os.path.join(ms_foler,ms_file_list[3]))
        outfile_580 = os.path.join(ms_resize_folder,ms_file_list[3].replace('.jpg','_resize.jpg'))
        out = imm_580.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_580)
        self.nm580.setPixmap(QPixmap(outfile_580))
        self.nm580.resize(100, 100)
        
        imm_600 = Image.open(os.path.join(ms_foler,ms_file_list[4]))
        outfile_600 = os.path.join(ms_resize_folder,ms_file_list[4].replace('.jpg','_resize.jpg'))
        out = imm_600.resize((256,192),Image.ANTIALIAS) #resize image with high-quality
        out.save(outfile_600)
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
        def cut_image(path,newpath):
            image_list = os.listdir(path)
            if '.DS_Store' in image_list:
                image_list.remove('.DS_Store')
            for ms in image_list:
                img = Image.open(path+f"{ms}")
                try:
                    os.mkdir(newpath)
                except:
                    pass
                for high in range(8):
                    for length in range(10):
                        cropped = img.crop((length*96, high*96,(length+1)*96 , (high+1)*96))  # (left, upper, right, lower)
                        cropped.save(newpath+f"{ms.replace('.jpg','')}_{high+1}_{length+1}.jpg")
                    cropped_extra = img.crop((928, high*96,1024 , (high+1)*96))  # (left, upper, right, lower)
                    cropped_extra.save(newpath+f"{ms.replace('.jpg','')}_{high+1}_11.jpg")

        def set_seperate_folder(path,newpath):

            train_list=os.listdir(path)
            try:
                os.mkdir(newpath)
            except:
                pass
            folder_list= os.listdir(newpath)
            if '.DS_Store' in train_list:
                train_list.remove('.DS_Store')
            for file in train_list:
                folder_name =file.split('_')[-2]+'_'+ file.split('_')[-1].replace('.jpg','')

                if  folder_name in folder_list:
                    shutil.copyfile(path+file,newpath+folder_name+'/'+file)
                elif folder_name not in folder_list:
                    try:
                        os.mkdir(newpath+folder_name)
                    except:
                        pass
                    shutil.copyfile(path+file,newpath+folder_name+'/'+file)
        def Concat_RGB_MS(MS_data,list):
            data_path = 'MultiSpectral/RGBResize'
            temp = np.zeros((len(list),96,96,7))
            for i in range(len(list)):
                img = Image.open(os.path.join(data_path, list[i] + '.png'))
                img = np.asarray(img.convert('RGB'))
                RGB = np.asarray(img)
                temp[i] = np.concatenate((RGB, np.squeeze(MS_data[i])), axis=-1)
            return temp

        def MSInterpolation(img):
            weight1 = [0.5,0,0.5,0]
            weight2 = [0.5,0.5,0,0]
            weight3 = [0,0.5,0.5,0]
            weight4 = [0.5,0,0,0.5]
            temp = np.zeros((len(img),96,96,4))
            for i in range(0,4):
                temp[:,:,:,i]=weight1[i]*img[:,:,:,0]+weight2[i]*img[:,:,:,1]+weight3[i]*img[:,:,:,2]+weight4[i]*img[:,:,:,3]
            img = np.concatenate((img,temp),axis=-1)
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

            X_val = []  # validation data
            temp_path = os.path.join(data_path)

            test_list = sorted(os.listdir(temp_path))
            if ".DS_Store" in test_list:
                test_list.remove(".DS_Store")

            test_list.sort(key=lambda x:(int(x.split('_')[0]),int(x.split('_')[-1].replace('.jpg',''))))
    
            if len(test_list)!=0:
                for nn in test_list:
                    temp_path1 = temp_path + '/' + nn
                    img_list = sorted(os.listdir(temp_path1))
                    single_train = []
                    for file in img_list:
                        img = imread(temp_path1+'/'+file)
                        img = np.dot(img, [0.299, 0.587, 0.114])
                        single_train.append(img)
                    single_train = np.transpose(single_train,(1,2,0))
                    X_val.append(single_train)
                X_val = np.asarray(X_val)
                X_val = X_val[:,:,:,1:6]

        #channel normalization
                max_channel = np.max(np.max(np.max(X_val,axis=0),axis=0),axis=0)
                min_channel = np.min(np.min(np.min(X_val,axis=0),axis=0),axis=0)
                for i in range(len(X_val)):
                    for j in range(X_val.shape[3]):
                        X_val[i, :, :, j] = (X_val[i, :, :, j] - min_channel[j]) / (max_channel[j] - min_channel[j]) * 255
            return X_val
#set path*********************************************************    
        inputfile_foler=self.inputFilePath.replace('.jpg','')
        try:
            os.mkdir(inputfile_foler)
        except:
            pass
        ms_path=inputfile_foler+'/'
        cutted_path=os.path.join(os.path.dirname(sys.argv[0]), 'allimages/')      
        seperated_path=os.path.join(os.path.dirname(sys.argv[0]), 'seperate/')
        load_model_path=os.path.join(os.path.dirname(sys.argv[0]), "saved_models/model_3.h5")
#set path*********************************************************    

#cut all all different ms images to one folder        
        cut_image(ms_path,cutted_path)
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

        np.save('x_val_3DUnet_gui.npy', X_val)
        model = Unet_Lightweight(numClass,NC)
        model_choice='_new_all_2_'#'_'#_new_  #_new_all_
        model.load_weights(load_model_path)
        x_test = np.load('x_val_3DUnet_gui.npy')
        try :
            os.mkdir(os.path.join(os.path.dirname(sys.argv[0]), 'pred_image'))
            
        except:
            pass
        predicted_path=os.path.join(os.path.dirname(sys.argv[0]), 'pred_image/')
        
        for imm in range(88):
            pred_test=model.predict(np.expand_dims(x_test[imm],axis=0),verbose=1)
            pred_test_t=pred_test.argmax(axis=-1)
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

        IMAGES_PATH = predicted_path 
        IMAGES_FORMAT = ['.jpg', '.JPG']  
        IMAGE_SIZE = 96 
        IMAGE_ROW = 8  
        IMAGE_COLUMN = 11
        cccc=load_model_path.split('/')[-1].replace('.h5','')
        self.IMAGE_SAVE_PATH = os.path.join(os.path.dirname(sys.argv[0]), 'final.jpg') 
        
        image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if  
        os.path.splitext(name)[1] == item]  
        image_names = sorted(os.listdir(IMAGES_PATH))
        image_names.remove("GUI_2.jpg")
        if ".DS_Store" in image_names:
            image_names.remove(".DS_Store")
        image_names.sort(key=lambda x:(int(x.split('.')[0])))
  
# def merge function 
        def image_compose():  
            to_image = Image.new('RGB', (1024, 768)) 
            for y in range(1, IMAGE_ROW + 1):  
                for x in range(1, IMAGE_COLUMN + 1):  
                    from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1])
           
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))  
            return to_image.save(self.IMAGE_SAVE_PATH)  
        image_compose()   
        try:
            shutil.rmtree(IMAGE_SIZE)
        except:
            pass
        predicted_file=self.IMAGE_SAVE_PATH
        self.predicted.setPixmap(QPixmap(predicted_file))
        self.predicted.resize(100, 100)
        imm_pre = Image.open(predicted_file)
        outfile_pre = predicted_file.replace('.jpg','_resize.jpg')
        out_pre = imm_pre.resize((512,384),Image.ANTIALIAS) #resize image with high-quality
        out_pre.save(outfile_pre)
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
