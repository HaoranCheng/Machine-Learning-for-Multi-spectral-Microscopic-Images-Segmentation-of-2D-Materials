import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from scipy.io import loadmat
import matplotlib.image as mpimg
from scipy import ndimage, misc
from imgaug import augmenters as iaa
import imgaug as ia
from PIL import Image, ImageOps
import tensorflow as tf

# 把图片分割成size大小的patch
def get_patches(img_arr, size=48, stride=48):

    patches_list = []
    overlapping = 0

    if size % stride != 0:
        raise ValueError("size % stride must be equal 0")
    if stride != size:
        overlapping = (size // stride) - 1
    if img_arr.ndim == 4:
        i_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(i_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[
                        i * stride: i * stride + size,
                        j * stride: j * stride + size,
                        ]
                    )

    else:
        raise ValueError("img_arr.ndim must be equal 4")

    return np.stack(patches_list)

# 把分割后的patch重组成大图
def reconstruct_from_patches(img_arr, org_img_size, stride=None, size=None):
    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    nm_layers = img_arr.shape[3]

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    total_nm_images = img_arr.shape[0] // (i_max ** 2)
    nm_images = img_arr.shape[0]

    averaging_value = size // stride
    images_list = []
    kk = 0
    for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], nm_layers), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for layer in range(nm_layers):
                    img_bg[
                    i * stride: i * stride + size,
                    j * stride: j * stride + size,
                    layer,
                    ] = img_arr[kk, :, :, layer]

                kk += 1


        images_list.append(img_bg)

    return np.stack(images_list)



# 把普通的label segmentation转换成onehot-label
# onehot-label 的channel数对应class的数量
def convert_to_onehot(label,numClass):
    one_hot=np.zeros((1,label.shape[0],label.shape[1],numClass),dtype=np.float32)
    for i in range (numClass):
        one_hot[0,:, :, i][label == i] = 1

    return one_hot



# 中值滤波模块
# 对分割patch的预测结果使用中值滤波可消除patch边缘的干扰
def median_f(img,size):
    for i in range(img.shape[0]):
        for j in range(img.shape[3]):
            img[i,:,:,j]=ndimage.median_filter(img[i,:,:,j], size)

    return img

# 对数据做augmentation
# 所有数据随机旋转aug=3次，旋转角度[0,90]之间
# 所有数据水平翻转一次

def load_data_aug(x_train, y_train,aug=4,channels=8,num_class=3,size=48):
    imgs=[]
    labels=[]
    num=x_train.shape[0]
    #num = 1
    for i in range(aug):
        for j in range(num):
            t = np.random.rand() * 90
            x_train_tmp1=x_train[j,:,:,:]
            y_train_tmp1=y_train[j,:,:,:]
            rotate = iaa.Affine(rotate=(t, t))
            x_train_rotate = rotate.augment_image(x_train_tmp1)
            y_train_rotate = rotate.augment_image(y_train_tmp1)
            imgs.append(x_train_rotate)
            labels.append(y_train_rotate)

    for k in range(num):
        x_train_tmp2 = x_train[k, :, :, :]
        y_train_tmp2 = y_train[k, :, :, :]
        imgs.append(x_train_tmp2)
        labels.append(y_train_tmp2)
        flip = iaa.Fliplr(1)
        x_train_flip = flip.augment_image(x_train_tmp2)
        y_train_flip = flip.augment_image(y_train_tmp2)
        imgs.append(x_train_flip)
        labels.append(y_train_flip)
    #cutmix
    # x_train_cutmix, y_train_cutmix = cutmix(x_train[:3],y_train[:3],aug=2)#aug必须小于8
    # plt.figure
    # plt.subplot(1,2,1)
    # plt.imshow(x_train[0])
    # plt.subplot(1,2,2)
    # plt.imshow(imgs[0])
    # #
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    #print(np.shape(x_train_cutmix),np.shape(y_train_cutmix))

    #如果要使用cutmix，使用以下两行
    # imgs = np.concatenate((imgs,x_train_cutmix),axis = 0)
    # labels = np.concatenate((labels,y_train_cutmix),axis = 0)
    return np.stack(imgs),np.stack(labels)


def cutmix(image, mask, aug, PROBABILITY=1.0):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    DIM = 96

    imgs = []
    masks = []
    for j in range(aug):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, aug), tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
        WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        # MAKE CUTMIX IMAGE
        one = image[j, ya:yb, 0:xa, :]
        two = image[k, ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
        imgs.append(img)
        # MAKE CUTMIX masks
        ein = mask[j, ya:yb, 0:xa, :]
        zwei = mask[k, ya:yb, xa:xb, :]
        drei = mask[j, ya:yb, xb:DIM, :]
        mitten = tf.concat([ein, zwei, drei], axis=1)
        mk = tf.concat([mask[j, 0:ya, :, :], mitten, mask[j, yb:DIM, :, :]], axis=0)
        masks.append(mk)

    return np.asarray(imgs), np.asarray(masks)



# 将binary的mask 转换成rgba的彩图
def mask_to_rgba(mask, color="red"):


    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)
    elif color == "gray": #background
        return np.stack((ones/2, ones/2, ones/2, ones), axis=-1)
    elif color == "royalblue": #1.layer
        return np.stack((ones/256*61, ones/256*165, ones/256*217, ones), axis=-1)
    elif color == "grassgreen": #2.layer
        return np.stack((ones/256*115, ones/256*191, ones/256*184, ones), axis=-1)
    elif color == "darkblue": #3.layer
        return np.stack((ones/256*35, ones/256*100, ones/256*170, ones), axis=-1)
    elif color == "gold": #4.layer
        return np.stack((ones/256*254, ones/256*198, ones/256*1, ones), axis=-1)
    elif color == "orange": #5.layer
        return np.stack((ones/256*234, ones/256*115, ones/256*23, ones), axis=-1)

# x_train=np.load('xT_RGB.npy')
# # # # #x_train_patch=get_patches(x_train)
# # # #
# y_train=np.load('yT_RGB.npy')
# # #
# x_train_aug,y_train_aug=load_data_aug(x_train,y_train,aug = 1)
