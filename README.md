# Machine-Learning-for-Multi-spectral-Microscopy-Images-Segmentation-of-2D-Materials
## Introduction

This is a public repository for deep learning-based accurate segmentation of 2D materials. The networkstructure were mainly developed by Mr. Zhuo Shi and optimized by Mr. Haoran Cheng (Email: haoran.cheng111@gmail.com), and were further polished by Mr. Hongwei Li (Email: hongwei.li@tum.de) and Mr. Xingchen Dong (xingchen.dong@tum.de).
<img width="797" alt="Screen Shot 2022-01-23 at 13 54 38" src="https://user-images.githubusercontent.com/33370630/150679356-1e9ae4af-181f-4451-b962-7fde181adfff.png">


<b>Input</b>: hyperspectral images (2+1 D) and RGB images (2D) \
<b>Output</b>: multi-class segmentation map of 2D materials \
<b>Model</b>: a two-stream convolutional neural network that fuses the dual-modality information \
<b>Key components</b>: 3D convolution, 2D convolution, 'Squeeze and Excitation' block \
<b>Loss function</b>: Dice-coefficient loss \
<b>Data augmentation</b>: random rotation, randomly cropping and randomly flipping 

The key architecture is defined in:
```
model.py
```
where the 3D and 2D features after convolutional layers and 'Squeeze and Excitation' blocks are fused into one network and trained in an end-to-end manner. We also provide the codes of different fusion strategies in *model.py* if you wish to compare them. 

![model](https://user-images.githubusercontent.com/33370630/150677430-8e62eb83-d8c7-486d-b8fa-869474fadced.png)
<img width="1326" alt="Screen Shot 2022-01-23 at 12 10 27" src="https://user-images.githubusercontent.com/33370630/150677708-a1a8797b-502b-4d0a-b229-8cce65333474.png">

How to train model: 
Input:There are 6 different wavelengths filters in the laboratory, the data shows below:
<img width="240" alt="Screen Shot 2022-01-23 at 13 26 40" src="https://user-images.githubusercontent.com/33370630/150678313-b7130cff-49bd-4525-838a-ee094eeb016d.png">
<img width="265" alt="Screen Shot 2022-01-23 at 13 26 47" src="https://user-images.githubusercontent.com/33370630/150678317-88926546-89dc-4f1b-ae2d-cc7cd68e3b61.png">
First, after the preprocess and screen process of original 1024*768 RGB images acquired from the laboratory, we put all these cutted images of same region in the same folder.
Secondly: Run MSTgray.py to prepare mix all the multispectral images in to one image and produce the npy file for training process.
Next: Run the K_fold_CV.py to train these images.
Finally: Run predict_withCCA.py to show the prediction of each layer.

GUI:
After optimized data set and performance of our model,



## How to train the network? 
First, you will need to prepare your dataset, following the steps introduced in the manuscript. \
Specifically, in our work the dimensions of the inputs are <b>[96, 96, 221, 1]</b> and <b>[96, 96, 3]</b> for hyperspectral images and RGB images (3 channels) respectively. To download the dataset and get to know the details, you can read the [description](https://github.com/hongweilibran/DALM/blob/main/ReadME_dataset.md) here.
Secondly, prepare your hardware and install requirements. GPU with 12GB or 24GB with cuda version 9.1 all work well. Then please install requirements via

```
pip install -r requirements.txt
```
Thirdly, start to train the network with demo codes named 'train_net.py' via:
```
python train_net.py
```
## Inference stage

When the training is done, the model and training curve are saved. 
Then you can have a look at the results via:  
```
python test_net.py
```
Then you will get some numbers of the evaluation metrics include Dice scores and Hausdorff distance.






