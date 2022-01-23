# Machine Learning for Multi-spectral Microscopic Images Segmentation of 2D Materials
## Introduction

This is a public repository for deep learning-based segmentation of 2D materials. The networkstructure were mainly developed by Mr. Zhuo Shi (ge75cam@mytum.de)and optimized by Mr. Haoran Cheng (Email: haoran.cheng111@gmail.com), and were further polished by Mr. Hongwei Li (Email: hongwei.li@tum.de) and Mr. Xingchen Dong (xingchen.dong@tum.de).
<img width="752" alt="Screen Shot 2022-01-23 at 13 58 33" src="https://user-images.githubusercontent.com/33370630/150679544-cec1f19d-e0ba-4515-b9df-b4bc5140ed16.png">


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

![model](https://user-images.githubusercontent.com/33370630/150677430-8e62eb83-d8c7-486d-b8fa-869474fadced.png)


## How to train network？
<b>Input</b>: There are 6 different wavelengths filters in the laboratory(500nm,520nm, 540nm, 56nm, 580nm, 600nm), the data shows below:
(500nm will not be used in the further training process, since it's too dark)

First, you will need to prepare your dataset. After the preprocess and screen process of original 1024*768 RGB images acquired from the laboratory, we put all these cutted images (<b>[96, 96, 3]</b> RGB images with 3 channels) of same region in the same folder.

<img width="240" alt="Screen Shot 2022-01-23 at 13 26 40" src="https://user-images.githubusercontent.com/33370630/150678313-b7130cff-49bd-4525-838a-ee094eeb016d.png"><img width="265" alt="Screen Shot 2022-01-23 at 13 26 47" src="https://user-images.githubusercontent.com/33370630/150678317-88926546-89dc-4f1b-ae2d-cc7cd68e3b61.png"><img width="313" alt="Screen Shot 2022-01-23 at 15 24 36" src="https://user-images.githubusercontent.com/33370630/150683177-53f10ae8-c92d-481f-b5cc-655aa5bde655.png">


Secondly: Run ```MSTgray.py``` to prepare and mix all the images with different wavelengths into multispectral image and save the npy file for training process.
Next: Run the ```K_fold_CV.py``` to train these images.
Finally: Run ```predict_withCCA.py``` to show the prediction of each layer.

After the processing them into multispectral images, To download the dataset and get to know the details, you can read the [description](https://github.com/hongweilibran/DALM/blob/main/ReadME_dataset.md) here.

## GUI
After optimized data set and performance of our model, a GUI system is built to predict large images(1024*768).
Run the ```GUI_chr.py```file to start the GUI.
Or use ```pyinstaller``` to pack it up as executable file.

<img width="1326" alt="Screen Shot 2022-01-23 at 12 10 27" src="https://user-images.githubusercontent.com/33370630/150677708-a1a8797b-502b-4d0a-b229-8cce65333474.png">

Prepare multispectral data for the GUI: ```Load Image``` will load the RGB image, ```Load MS_Image``` will show the images with different wavelengths.
```Predict``` will predict the flakes.


<img width="258" alt="Screen Shot 2022-01-23 at 15 15 55" src="https://user-images.githubusercontent.com/33370630/150682775-4be9fcf8-7815-41f4-9f43-0408188511df.png"><img width="234" alt="Screen Shot 2022-01-23 at 15 16 07" src="https://user-images.githubusercontent.com/33370630/150682783-c029800b-6a5f-4de3-a37b-748b497d50ef.png">






