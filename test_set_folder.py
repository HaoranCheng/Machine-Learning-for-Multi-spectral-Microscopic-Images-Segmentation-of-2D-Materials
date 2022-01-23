import os
import shutil

#将9792张图片分类提取并创建文件夹
'''
path= '/Users/haoran/Desktop/newregions/'
newpath= '/Users/haoran/Desktop/data/'
train_list=os.listdir(path)
#print(train_list)
folder_list= os.listdir(newpath)
#print(folder_list)
if '.DS_Store' in train_list:
    train_list.remove('.DS_Store')
for file in train_list:
    folder_name =file.split('_')[0]+'_'+ file.split('_')[2]
    if  folder_name in folder_list:
        shutil.copyfile(path+file,newpath+folder_name+"/"+file)
        #shutil.move(path+file,newpath+folder_name)
    elif folder_name not in folder_list:
        try:
            os.mkdir(newpath+folder_name)
        except:
            print('exists')
        shutil.copyfile(path+file,newpath+folder_name+"/"+file)
        print('New folder created')'''

def set_seperate_folder(path,newpath):
    #path= '/Users/haoran/Desktop/newregions/'
    #newpath= '/Users/haoran/Desktop/data/'
    train_list=os.listdir(path)
#print(train_list)
    folder_list= os.listdir(newpath)
#print(folder_list)
    if '.DS_Store' in train_list:
        train_list.remove('.DS_Store')
    for file in train_list:
        folder_name =file.split('_')[0]+'_'+ file.split('_')[2]
        if  folder_name in folder_list:
            shutil.copyfile(path+file,newpath+folder_name+"/"+file)
        #shutil.move(path+file,newpath+folder_name)
        elif folder_name not in folder_list:
            try:
                os.mkdir(newpath+folder_name)
            except:
                print('exists')
            shutil.copyfile(path+file,newpath+folder_name+"/"+file)
            print('New folder created')



