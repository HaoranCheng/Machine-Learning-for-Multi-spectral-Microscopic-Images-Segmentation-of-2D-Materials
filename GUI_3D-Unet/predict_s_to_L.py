from PIL import Image
import os
import shutil
from tkinter.filedialog import test
import numpy as np
from cv2 import imread
import matplotlib.pyplot as plt
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
#opt= mpimg.imread('newMScrop/Label/2_74.jpg')
from MSToGray import load_MS
#预测大图的函数，讲大图裁剪为小图并预测后合并为大图

#裁剪并且，每张小图设置单独文件夹
def cut_image(path,newpath):
    print('Original Image:',os.listdir(path))
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
            print('already exist')
        for high in range(8):
            for length in range(10):
                cropped = img.crop((length*96, high*96,(length+1)*96 , (high+1)*96))  # (left, upper, right, lower)
                cropped.save(newpath+f"{ms.strip('.jpg')}_{high+1}_{length+1}.jpg")
            cropped_extra = img.crop((928, high*96,1024 , (high+1)*96))  # (left, upper, right, lower)
            cropped_extra.save(newpath+f"{ms.strip('.jpg')}_{high+1}_11.jpg")

def set_seperate_folder(path,newpath):
    #path= '/Users/haoran/Desktop/newregions/'
    #newpath= '/Users/haoran/Desktop/data/'
    train_list=os.listdir(path)
#print(train_list)
    try:
        os.mkdir(newpath)
    except:
            print('newpath exists')
    folder_list= os.listdir(newpath)
    #print(folder_list)
    if '.DS_Store' in train_list:
        train_list.remove('.DS_Store')
    for file in train_list:
        folder_name =file.split('_')[-2]+'_'+ file.split('_')[-1].strip('.jpg')
        #print(folder_name)
        if  folder_name in folder_list:
            shutil.copyfile(path+file,newpath+folder_name+'/'+file)
        #shutil.move(path+file,newpath+folder_name)
        elif folder_name not in folder_list:
            try:
                os.mkdir(newpath+folder_name)
            except:
                print('exists')
            #print(folder_name)
            shutil.copyfile(path+file,newpath+folder_name+'/'+file)
            print('New folder created')
            
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
    print("Original:"+str(np.shape(img)))
    weight1 = [0.5,0,0.5,0]
    weight2 = [0.5,0.5,0,0]
    weight3 = [0,0.5,0.5,0]
    weight4 = [0.5,0,0,0.5]
    temp = np.zeros((len(img),96,96,4))
    for i in range(0,4):
        temp[:,:,:,i]=weight1[i]*img[:,:,:,0]+weight2[i]*img[:,:,:,1]+weight3[i]*img[:,:,:,2]+weight4[i]*img[:,:,:,3]
    img = np.concatenate((img,temp),axis=-1)
    print(np.shape(img))
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
    # print(temp_path)
    test_list = sorted(os.listdir(temp_path))
    if ".DS_Store" in test_list:
        test_list.remove(".DS_Store")
    #train_list = sorted(os.listdir(temp_path))
    test_list.sort(key=lambda x:(int(x.split('_')[0]),int(x.split('_')[-1].strip('.jpg'))))
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
             # print(np.max(label))
             # print(np.shape(label))
             # print(np.shape(label))
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
ms_path=os.getcwd()+"/newregion1/"
cutted_path=os.getcwd()+"/newregion1/allimages/"#mkdir
seperated_path=os.getcwd()+"/newregion1/sepeate/"#mkdir
load_model_path=os.getcwd()+"/saved_models/chr/model_3lsame_gamma_aug_1.h5"
#*********************************************************      

#裁剪所有所有nm的大图片并放在一个文件夹           
#cut_image(os.getcwd()+"/GUI_cheng/newregion1",os.getcwd()+"/GUI_cheng/newregion1/allimages")       
#set_seperate_folder(os.getcwd()+"/GUI_cheng/newregion1/allimages/",os.getcwd()+"/GUI_cheng/newregion1/sepeate/")
cut_image(ms_path,cutted_path)#seperate 的path是cutimgae的newpath
set_seperate_folder(cutted_path,seperated_path)

X_val= load_MS(seperated_path)
#load_MS的path是seperate的路径)
print("Val"+str(np.shape(X_val)))
np.save('x_val_3DUnet_gui.npy', X_val)



#预测小图片并放在precicted_path
model = Unet_Lightweight(numClass,NC)
model_choice='_new_all_2_'#'_'#_new_  #_new_all_
model.load_weights(load_model_path)
x_test = np.load('x_val_3DUnet_gui.npy')

x_test_DR = Perform_DR(method,NC,x_test)
print('npy file.shape:',x_test_DR.shape)
try :
    os.mkdir(os.getcwd()+'/pred_image')
except:
    print('existed')
predicted_path=os.getcwd()+'/pred_image/'
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
    ax14 = fig.add_subplot(3,6,14)    #"gray","royalblue","grassgreen","darkblue","gold","orange"
    plt.imshow(mask_to_rgba(pred_0,"gray"))
    plt.imshow(mask_to_rgba(pred_1,"royalblue"))
    plt.imshow(mask_to_rgba(pred_2,"grassgreen"))
    plt.imshow(mask_to_rgba(pred_3,"darkblue"))
    plt.imshow(mask_to_rgba(pred_4,"gold"))
    plt.imshow(mask_to_rgba(pred_5,"orange"))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
#ax14.set_title("true-mask",fontsize=8)

# if NC == 251:
#     plt.savefig('./Quantitive Analysis(Without DR).jpg')
# else:
#     plt.savefig('./Quantitive Analysis'+str(NC)+'.jpg')
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
IMAGE_SAVE_PATH = os.getcwd()+'/final1111.jpg' # 图片转换后的地址  

# 获取图片集地址下的所有图片名称  
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if  
os.path.splitext(name)[1] == item]  
#image_names=os.listdir
image_names = sorted(os.listdir(IMAGES_PATH))
image_names.remove("GUI_2.jpg")
if ".DS_Store" in image_names:
    image_names.remove(".DS_Store")
image_names.sort(key=lambda x:(int(x.split('.')[0])))
print(image_names)
  
# 定义图像拼接函数  
def image_compose():  
    to_image = Image.new('RGB', (1024, 768)) #创建一个新图  
# 循环遍历，把每张图片按顺序粘贴到对应位置上  
    for y in range(1, IMAGE_ROW + 1):  
        for x in range(1, IMAGE_COLUMN + 1):  
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1])
            '''
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(  
            (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)  '''
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))  
    return to_image.save(IMAGE_SAVE_PATH) # 保存新图  
image_compose() #调用函数  
