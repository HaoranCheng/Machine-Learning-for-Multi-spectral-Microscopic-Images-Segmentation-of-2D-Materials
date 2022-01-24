# Machine Learning for Multi-spectral Microscopic Images Segmentation of 2D Materials
## Introduction

This is a public repository for deep learning-based segmentation of 2D materials. The networkstructure were mainly developed by Mr. Zhuo Shi (ge75cam@mytum.de)and optimized by Mr. Haoran Cheng (Email: haoran.cheng111@gmail.com), and were further polished by Mr. Hongwei Li (Email: hongwei.li@tum.de) and Mr. Xingchen Dong (xingchen.dong@tum.de).
<img width="892" alt="Screen Shot 2022-01-24 at 12 06 19" src="https://user-images.githubusercontent.com/33370630/150771830-7fb5d4f3-819b-4a5e-8ede-831c55f7ac12.png">


<b>Input</b>: multi-spectral images (2+1 D) and RGB images (2D) \
<b>Output</b>: multi-class segmentation map of 2D materials \
<b>Model</b>: The Lightweight 3D U-net\
<b>Loss function</b>: Dice-coefficient loss or the focal Tversky loss\
<b>Data augmentation</b>: random rotation, randomly cropping and randomly flipping 

The key architecture is defined in:
```
model.py
```
where the 3D and 2D features after convolutional layers and 'Squeeze and Excitation' blocks are fused into one network and trained in an end-to-end manner. We also provide the codes of different fusion strategies in ```model.py``` if you wish to compare them. 



## How to train network？
<b>Input</b>: There are 6 different wavelengths filters in the laboratory(500nm,520nm, 540nm, 56nm, 580nm, 600nm), the data shows below:
(500nm will not be used in the further training process, since it's too dark)

First, you will need to prepare your dataset. After the preprocess and screen process of original 1024*768 RGB images acquired from the laboratory, we put all these cutted images (<b>[96, 96, 3]</b> RGB images with 3 channels) of same region in the same folder.


Secondly: Run ```MSTgray.py``` to prepare and mix all the images with different wavelengths into multispectral image and save the npy file for training process.
Next: Run the ```K_fold_CV.py``` to train these images.
Finally: Run ```predict_withCCA.py``` to show the prediction of each layer.

## GUI
After optimized data set and performance of our model, a GUI system is built to predict large images(1024*768).
Run the ```GUI_chr.py```file to start the GUI.
Or use ```pyinstaller``` to pack it up as executable file.
```Load Image``` will load the RGB image, ```Load MS_Image``` will show the images with different wavelengths.
```Predict``` will predict the flakes.
<img width="1326" alt="Screen Shot 2022-01-23 at 12 10 27" src="https://user-images.githubusercontent.com/33370630/150677708-a1a8797b-502b-4d0a-b229-8cce65333474.png">

 




