# -*- coding: utf-8 -*-
"""
Created on 2019-3-12
@author: LeonShangguan
"""

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import io
import skimage.transform as trans
import glob
import os
import skimage.io as io
import cv2

white = [255, 255, 255]
black = [0, 0, 0]
red = [255, 0, 0]

COLOR_DICT = np.array([white, black, red])


def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
        new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif (np.max(img) > 1):
        img = img / 255.0
        mask = mask / 255.0
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="rgb",
                   mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(512, 512), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        print(img.shape)
        yield (img, mask)

def testGenerator(test_path, target_size=(512, 512), flag_multi_class=False, as_gray=False):
    for name in glob.glob(test_path + '*.jpg'):
        # img = io.imread(name)
        img = cv2.imread(name)
        img = img / 255.0
        # print(img.shape)
        img = trans.resize(img, target_size)
        # img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        # print(img.shape)
        yield img


def labelVisualize(num_class, color_dict, img):
    # print(img.shape)
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255.0


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        # img = cv2.resize(img, (420, 420))
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
        # cv2.imwrite(os.path.join(save_path, "%d_predict.png" % i), img)


def generate_overkill(path,image_start,image_end):
    train_x= cv2.imread(os.path.join(path,"%d.jpg"%(image_start)),-1)
    train_y=cv2.imread(os.path.join(path,"mask.jpg"),-1)

    img_y=cv2.imread(os.path.join(path,"mask.jpg"),-1)


    for i in range (image_start+1,image_end):
        img = cv2.imread(os.path.join(path,"%d.jpg"%(i)),-1)
        train_x=np.concatenate((train_x,img),axis=0)
        train_y=np.concatenate((train_y,img_y),axis=0)

    train_x=train_x.reshape((-1,512,512,3))
    train_x = train_x.astype('float32')
    train_x /= 255

    train_y=train_y.reshape((-1,512,512,3))
    train_y = train_y.astype('float32')
    train_y /= 255

    return train_x,train_y


def generate_train(path,image_start,image_end,property):
    train_x= cv2.imread(os.path.join(path,"%d.png"%(image_start)),-1)

    for i in range (image_start+1,image_end):
        img = cv2.imread(os.path.join(path,"%d.png"%(i)),-1)
        train_x=np.concatenate((train_x,img),axis=0)
    
    # for i in range (image_start,image_end):
    #     img = cv2.imread(os.path.join(path,"%d.png"%(i)),-1)
    #     train_x=np.concatenate((train_x,img),axis=0)
    # for i in range (image_start,image_end):
    #     img = cv2.imread(os.path.join(path,"%d.jpg"%(i)),-1)
    #     train_x=np.concatenate((train_x,img),axis=0)

    # for i in range (image_start,image_end):
    #     img = cv2.imread(os.path.join(path,"%d.jpg"%(i)),-1)
    #     train_x=np.concatenate((train_x,img),axis=0)

    train_x=train_x.reshape((-1,512,512,3))
    train_x = train_x.astype('float32')
    train_x /= 255
    return train_x