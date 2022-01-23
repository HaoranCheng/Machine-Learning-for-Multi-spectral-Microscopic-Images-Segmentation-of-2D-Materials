import numpy as np
import matplotlib
import os
from PIL import Image

def img_seg(dir):
    files = os.listdir(dir)
    for file in files:
        a, b = os.path.splitext(file)
        print(a,b)
        img = Image.open(os.path.join(dir + "/" + file))
        hight, width = img.size
        w = 96*2    #切割成812*812
        id = 1
        i = 0
        while (i + w <= hight):
            j = 0
            while (j + w <= width):
                new_img = img.crop((i, j, i + w, j + w))
                rename = os.getcwd()+"/region_cheng/newregion8/"
                new_img = new_img.resize((96,96), Image.ANTIALIAS)
                new_img.save(rename + a + "_" + str(id) + b)
                id += 1
                j += 50   #滑动步长
            i = i + 50
#find ./ -name ".DS_Store" -depth -exec rm {} \;

if __name__ == '__main__':
    path = "./2"
    img_seg(os.getcwd()+"/XingchenDong/newregion8")
