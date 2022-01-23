import os
import shutil
#这个py文件用来选出3层的图片文件夹
#移动的文件 /Users/haoran/Desktop/untitled folder/train_data_1375+74/1007_110
path= '/Users/haoran/Desktop/untitled folder/train_data_1375+74/'
tri_path='/Users/haoran/Desktop/untitled folder/3layer/'#trilayer path
move_path='/Users/haoran/Desktop/untitled folder/3layer_multi/'

tri_list= os.listdir(tri_path)
#需要删除的图片的文件夹
print(tri_list)
if '.DS_Store' in tri_list:
    tri_list.remove('.DS_Store')
try:
    
    for sigle_tri in tri_list:
    #print(de)
    #print(de.split('.')[0])
    #os.remove(path+de)
        print("移动的文件",path+sigle_tri.split('.')[0])
        orginal=path+sigle_tri.split('.')[0]
        #shutil.copytree(path+sigle_tri.split('.')[0],move_path)
        shutil.move(path+sigle_tri.split('.')[0],move_path)#1005_115_.jpg将jpg去掉并读取文件夹
except:
    print('error')
