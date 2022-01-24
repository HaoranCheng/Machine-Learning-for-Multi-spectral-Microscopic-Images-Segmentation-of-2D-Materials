import numpy as np
import os
from PIL import Image
from cv2 import imread
import matplotlib.pyplot as plt
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


def load_MS():
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
    data_path = 'newMScrop/cheng/'

    X_train = []  # training data
    y_train = []  # training label
    X_val = []  # validation data
    y_val = []  # validation label
    temp_path = os.path.join(data_path, 'crop','train')
    # print(temp_path)
    train_list = sorted(os.listdir(temp_path))
    train_list.sort(key=lambda x:(int(x.split('_')[0]),int(x.split('_')[-1])))
    print(train_list)
    for nn in train_list:
         temp_path1 = temp_path + '/' + nn
         img_list = sorted(os.listdir(temp_path1))
         
         single_train = []
         for file in img_list:
             print(file)
             # print(np.shape(np.array(img)))
             img = imread(temp_path1 + '/' + file)
             #print(np.shape(np.array(img)))
             img = np.dot(img, [0.299, 0.587, 0.114])
             single_train.append(img)
         single_train = np.transpose(single_train,(1,2,0))#1*96*96*5
         X_train.append(single_train)#n*96*96*5
         #print(np.shape(X_train))
         label = np.asarray(Image.open(os.path.join(data_path, 'Label',nn+'_Simple Segmentation.png')))
         one_hot_label = np.squeeze(convert_to_onehot(label, NUM_class))
         y_train.append(one_hot_label)
    X_train = np.array(X_train)
    X_train = X_train[:,:,:,1:6]
    y_train = np.array(y_train)
    # Light compensation
    Bg_after = Layer_extraction(X_train[0],y_train[0],0)
    #Bg_before = Layer_extraction(X_train[68],y_train[68],0)
    Bg_before = Layer_extraction(X_train[-1],y_train[-1],0)
    X_train[0:73] = X_train[0:73]*Bg_before/Bg_after
    X_train[74:] = X_train[74:] * Bg_before / Bg_after
    
    print(X_train.shape)
    #channel normalization
    max_channel = np.max(np.max(np.max(X_train,axis=0),axis=0),axis=0)
    #print("max_channel"+str(max_channel))
    min_channel = np.min(np.min(np.min(X_train,axis=0),axis=0),axis=0)

    for i in range(len(X_train)):
        for j in range(X_train.shape[3]):
            X_train[i,:,:,j] = (X_train[i,:,:,j]-min_channel[j])/(max_channel[j]-min_channel[j])*255

    temp_path = os.path.join(data_path, 'crop','test')
    # print(temp_path)
    test_list = sorted(os.listdir(temp_path))
    if len(test_list)!=0:
        for nn in test_list:
             temp_path1 = temp_path + '/' + nn
             img_list = sorted(os.listdir(temp_path1))
             single_train = []
             for file in img_list:
                 #print(file)
                 img = imread(temp_path1+'/'+file)
                 img = np.dot(img, [0.299, 0.587, 0.114])
                 single_train.append(img)
             single_train = np.transpose(single_train,(1,2,0))
             X_val.append(single_train)
             #print(np.shape(X_train))
             label = np.asarray(Image.open(os.path.join(data_path, 'Label', nn+'_Simple Segmentation.png')))
             # print(np.max(label))
             # print(np.shape(label))
             # print(np.shape(label))
             one_hot_label = np.squeeze(convert_to_onehot(label, NUM_class))
             #print(none_hot_label[:,:,3]))
             y_val.append(one_hot_label)
        #delete first channel，which is too dark
        X_val = np.asarray(X_val)
        X_val = X_val[:,:,:,1:6]
        y_val = np.asarray(y_val)

        #light compensation
        X_val[-2:] = X_val[-2:]*Bg_before/Bg_after
       

        #channel normalization
        max_channel = np.max(np.max(np.max(X_val,axis=0),axis=0),axis=0)
        min_channel = np.min(np.min(np.min(X_val,axis=0),axis=0),axis=0)
        for i in range(len(X_val)):
            for j in range(X_val.shape[3]):
                X_val[i, :, :, j] = (X_val[i, :, :, j] - min_channel[j]) / (max_channel[j] - min_channel[j]) * 255
    return X_train, y_train, X_val, y_val

def main():
  
    X_train,Y_train, X_val, Y_val = load_MS()
    print("Train"+str(np.shape(X_train)))
    print("Train"+str(np.shape(Y_train)))
    print("Val"+str(np.shape(X_val)))
    print("VAl_label"+str(np.shape(Y_val)))
    
    np.save('x_train_3DUnet_new.npy', X_train)
    np.save('y_train_3DUnet_new.npy', Y_train)
    np.save('x_val_3DUnet_new.npy', X_val)
    np.save('y_val_3DUnet_new.npy', Y_val)


if __name__ == "__main__":
    main()
