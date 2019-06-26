#from model import *
from data import *
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default="data/membrane/422_change_skip/bce_dice_loss_change_skip_50_1", type=str, help='path of the saved data')
parser.add_argument('--model_name', default="421_test_skip/bce_dice_loss_change_skip_50_1.hdf5", type=str, help='model_name')
parser.add_argument('--Epochs', default=50, type=int, help='Epochs')

from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

def My_unet_512(input_shape=(512, 512, 1),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024

    # down0b = Conv2D(8, (3, 3), padding='same')(inputs)
    # down0b = BatchNormalization()(down0b)
    # down0b = Activation('relu')(down0b)
    # down0b = Conv2D(8, (3, 3), padding='same')(down0b)
    # down0b = BatchNormalization()(down0b)
    # down0b = Activation('relu')(down0b)
    # down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    # up0b = UpSampling2D((2, 2))(up0a)
    # up0b = concatenate([down0b, up0b], axis=3)
    # up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    # up0b = BatchNormalization()(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    # up0b = BatchNormalization()(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    # up0b = BatchNormalization()(up0b)
    # up0b = Activation('relu')(up0b)

    # 1024


    s_down0b = Conv2D(8, (3, 3), padding='same')(inputs)

    # s_down0b = Conv2D(8, (3, 3), padding='same')(up0b)

    s_down0b = BatchNormalization()(s_down0b)
    s_down0b = Activation('relu')(s_down0b)
    s_down0b = Conv2D(8, (3, 3), padding='same')(s_down0b)
    s_down0b = BatchNormalization()(s_down0b)
    s_down0b = Activation('relu')(s_down0b)
    s_down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(s_down0b)
    # 512

    s_down0a = Conv2D(16, (3, 3), padding='same')(s_down0b_pool)

    s_down0a = concatenate([up0a, s_down0a], axis=3)

    s_down0a = BatchNormalization()(s_down0a)
    s_down0a = Activation('relu')(s_down0a)
    s_down0a = Conv2D(16, (3, 3), padding='same')(s_down0a)
    s_down0a = BatchNormalization()(s_down0a)
    s_down0a = Activation('relu')(s_down0a)
    s_down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(s_down0a)
    # 256

    s_down0 = Conv2D(32, (3, 3), padding='same')(s_down0a_pool)

    s_down0 = BatchNormalization()(s_down0)
    s_down0 = Activation('relu')(s_down0)
    s_down0 = Conv2D(32, (3, 3), padding='same')(s_down0)
    s_down0 = BatchNormalization()(s_down0)
    s_down0 = Activation('relu')(s_down0)
    s_down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(s_down0)
    # 128

    s_down1 = Conv2D(64, (3, 3), padding='same')(s_down0_pool)

    s_down1 = BatchNormalization()(s_down1)
    s_down1 = Activation('relu')(s_down1)
    s_down1 = Conv2D(64, (3, 3), padding='same')(s_down1)
    s_down1 = BatchNormalization()(s_down1)
    s_down1 = Activation('relu')(s_down1)
    s_down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(s_down1)
    # 64

    s_down2 = Conv2D(128, (3, 3), padding='same')(s_down1_pool)

    s_down2 = BatchNormalization()(s_down2)
    s_down2 = Activation('relu')(s_down2)
    s_down2 = Conv2D(128, (3, 3), padding='same')(s_down2)
    s_down2 = BatchNormalization()(s_down2)
    s_down2 = Activation('relu')(s_down2)
    s_down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(s_down2)
    # 32

    s_down3 = Conv2D(256, (3, 3), padding='same')(s_down2_pool)

    s_down3 = BatchNormalization()(s_down3)
    s_down3 = Activation('relu')(s_down3)
    s_down3 = Conv2D(256, (3, 3), padding='same')(s_down3)
    s_down3 = BatchNormalization()(s_down3)
    s_down3 = Activation('relu')(s_down3)
    s_down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(s_down3)
    # 16

    s_down4 = Conv2D(512, (3, 3), padding='same')(s_down3_pool)

    s_down4 = BatchNormalization()(s_down4)
    s_down4 = Activation('relu')(s_down4)
    s_down4 = Conv2D(512, (3, 3), padding='same')(s_down4)
    s_down4 = BatchNormalization()(s_down4)
    s_down4 = Activation('relu')(s_down4)
    s_down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(s_down4)
    # 8

    s_center = Conv2D(1024, (3, 3), padding='same')(s_down4_pool)
    s_center = BatchNormalization()(s_center)
    s_center = Activation('relu')(s_center)
    s_center = Conv2D(1024, (3, 3), padding='same')(s_center)
    s_center = BatchNormalization()(s_center)
    s_center = Activation('relu')(s_center)
    # center

    s_up4 = UpSampling2D((2, 2))(s_center)
    s_up4 = concatenate([s_down4, s_up4], axis=3)
    s_up4 = Conv2D(512, (3, 3), padding='same')(s_up4)
    s_up4 = BatchNormalization()(s_up4)
    s_up4 = Activation('relu')(s_up4)
    s_up4 = Conv2D(512, (3, 3), padding='same')(s_up4)
    s_up4 = BatchNormalization()(s_up4)
    s_up4 = Activation('relu')(s_up4)
    s_up4 = Conv2D(512, (3, 3), padding='same')(s_up4)
    s_up4 = BatchNormalization()(s_up4)
    s_up4 = Activation('relu')(s_up4)
    # 16

    s_up3 = UpSampling2D((2, 2))(s_up4)
    s_up3 = concatenate([s_down3, s_up3], axis=3)
    s_up3 = Conv2D(256, (3, 3), padding='same')(s_up3)
    s_up3 = BatchNormalization()(s_up3)
    s_up3 = Activation('relu')(s_up3)
    s_up3 = Conv2D(256, (3, 3), padding='same')(s_up3)
    s_up3 = BatchNormalization()(s_up3)
    s_up3 = Activation('relu')(s_up3)
    s_up3 = Conv2D(256, (3, 3), padding='same')(s_up3)
    s_up3 = BatchNormalization()(s_up3)
    s_up3 = Activation('relu')(s_up3)
    # 32

    s_up2 = UpSampling2D((2, 2))(s_up3)
    s_up2 = concatenate([s_down2, s_up2], axis=3)
    s_up2 = Conv2D(128, (3, 3), padding='same')(s_up2)
    s_up2 = BatchNormalization()(s_up2)
    s_up2 = Activation('relu')(s_up2)
    s_up2 = Conv2D(128, (3, 3), padding='same')(s_up2)
    s_up2 = BatchNormalization()(s_up2)
    s_up2 = Activation('relu')(s_up2)
    s_up2 = Conv2D(128, (3, 3), padding='same')(s_up2)
    s_up2 = BatchNormalization()(s_up2)
    s_up2 = Activation('relu')(s_up2)
    # 64

    s_up1 = UpSampling2D((2, 2))(s_up2)
    s_up1 = concatenate([s_down1, s_up1], axis=3)
    s_up1 = Conv2D(64, (3, 3), padding='same')(s_up1)
    s_up1 = BatchNormalization()(s_up1)
    s_up1 = Activation('relu')(s_up1)
    s_up1 = Conv2D(64, (3, 3), padding='same')(s_up1)
    s_up1 = BatchNormalization()(s_up1)
    s_up1 = Activation('relu')(s_up1)
    s_up1 = Conv2D(64, (3, 3), padding='same')(s_up1)
    s_up1 = BatchNormalization()(s_up1)
    s_up1 = Activation('relu')(s_up1)
    # 128

    s_up0 = UpSampling2D((2, 2))(s_up1)
    s_up0 = concatenate([s_down0, s_up0], axis=3)
    s_up0 = Conv2D(32, (3, 3), padding='same')(s_up0)
    s_up0 = BatchNormalization()(s_up0)
    s_up0 = Activation('relu')(s_up0)
    s_up0 = Conv2D(32, (3, 3), padding='same')(s_up0)
    s_up0 = BatchNormalization()(s_up0)
    s_up0 = Activation('relu')(s_up0)
    s_up0 = Conv2D(32, (3, 3), padding='same')(s_up0)
    s_up0 = BatchNormalization()(s_up0)
    s_up0 = Activation('relu')(s_up0)
    # 256

    s_up0a = UpSampling2D((2, 2))(s_up0)
    s_up0a = concatenate([s_down0a, s_up0a], axis=3)
    s_up0a = Conv2D(16, (3, 3), padding='same')(s_up0a)
    s_up0a = BatchNormalization()(s_up0a)
    s_up0a = Activation('relu')(s_up0a)
    s_up0a = Conv2D(16, (3, 3), padding='same')(s_up0a)
    s_up0a = BatchNormalization()(s_up0a)
    s_up0a = Activation('relu')(s_up0a)
    s_up0a = Conv2D(16, (3, 3), padding='same')(s_up0a)
    s_up0a = BatchNormalization()(s_up0a)
    s_up0a = Activation('relu')(s_up0a)
    # 512

    s_up0b = UpSampling2D((2, 2))(s_up0a)
    s_up0b = concatenate([s_down0b, s_up0b], axis=3)
    s_up0b = Conv2D(8, (3, 3), padding='same')(s_up0b)
    s_up0b = BatchNormalization()(s_up0b)
    s_up0b = Activation('relu')(s_up0b)
    s_up0b = Conv2D(8, (3, 3), padding='same')(s_up0b)
    s_up0b = BatchNormalization()(s_up0b)
    s_up0b = Activation('relu')(s_up0b)
    s_up0b = Conv2D(8, (3, 3), padding='same')(s_up0b)
    s_up0b = BatchNormalization()(s_up0b)
    s_up0b = Activation('relu')(s_up0b)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(s_up0b)

    model = Model(inputs=inputs, outputs=classify)

    model.compile('Adam', loss=weighted_bce_dice_loss, metrics=[iou_score])

    return model

def process(file_path,model_name,Epochs):
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='nearest')
    myGene = trainGenerator(5,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
    model=My_unet_512()
    #model = unet()
    # model = Unet('resnet152', input_shape=(None, None, 1), classes=2, encoder_weights=None)

    model.compile('Adam', loss=weighted_bce_dice_loss, metrics=[iou_score])


    model_checkpoint = ModelCheckpoint(model_name, monitor='loss',verbose=2, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=300,epochs=Epochs,callbacks=[model_checkpoint])


    testGene = testGenerator("data/membrane/test")
    results = model.predict_generator(testGene,30,verbose=1)
    saveResult(file_path,results)


def main():
    args = parser.parse_args(sys.argv[1:])
    return process(file_path=args.file_path,model_name=args.model_name,Epochs=args.Epochs)
 

if __name__ == "__main__":
    main()

   