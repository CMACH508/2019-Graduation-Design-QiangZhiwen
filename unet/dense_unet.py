#from model import *
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

def dense_block(input,size):
    output = Conv2D(size[0][1], (size[0][0], size[0][0]), padding='same')(input)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = Conv2D(size[1][1], (size[1][0], size[1][0]), padding='same')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = Conv2D(size[2][1], (size[2][0], size[2][0]), padding='same')(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output


def My_unet_512(input_shape=(512, 512, 1),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024


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

    # down1_1=dense_block(down0_pool,[[1,16],[3,16],[1,64]])
    # down1_2=dense_block(down1_1,[[1,16],[3,16],[1,64]])
    # down1_2_1=concatenate([down1_2, down1_1], axis=3)
    # down1=dense_block(down1_2_1,[[1,16],[3,16],[1,64]])
    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    


    # down2_1=dense_block(down1_pool,[[1,32],[3,32],[1,128]])
    # down2_2=dense_block(down2_1,[[1,32],[3,32],[1,128]])
    # down2_2_1=concatenate([down2_2, down2_1], axis=3)
    # down2=dense_block(down2_2_1,[[1,32],[3,32],[1,128]])
    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32


    down3_1=dense_block(down2_pool,[[1,64],[3,64],[1,256]])
    down3_2=dense_block(down3_1,[[1,64],[3,64],[1,256]])
    down3_2_1=concatenate([down3_2, down3_1], axis=3)
    down3=dense_block(down3_2_1,[[1,64],[3,64],[1,256]])
    # down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    # down3 = BatchNormalization()(down3)
    # down3 = Activation('relu')(down3)
    # down3 = Conv2D(256, (3, 3), padding='same')(down3)
    # down3 = BatchNormalization()(down3)
    # down3 = Activation('relu')(down3)
    #(32,32,256)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4_1=dense_block(down3_pool,[[1,128],[3,128],[1,512]])
    down4_2=dense_block(down4_1,[[1,128],[3,128],[1,512]])
    down4_2_1=concatenate([down4_2, down4_1], axis=3)
    down4_3=dense_block(down4_2_1,[[1,128],[3,128],[1,512]])
    down4_3_2_1=concatenate([down4_3, down4_2_1], axis=3)
    down4=dense_block(down4_3_2_1,[[1,128],[3,128],[1,512]])
    # down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    # down4 = BatchNormalization()(down4)
    # down4 = Activation('relu')(down4)
    # down4 = Conv2D(512, (3, 3), padding='same')(down4)
    # down4 = BatchNormalization()(down4)
    # down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8


    center_1=dense_block(down4_pool,[[1,256],[3,256],[1,1024]])
    center_2=dense_block(center_1,[[1,256],[3,256],[1,1024]])
    center_2_1=concatenate([center_2, center_1], axis=3)
    center_3=dense_block(center_2_1,[[1,256],[3,256],[1,1024]])
    center_3_2_1=concatenate([center_3, center_2_1], axis=3)
    center_4=dense_block(center_3_2_1,[[1,256],[3,256],[1,1024]])
    center_4_3_2_1=concatenate([center_4, center_3_2_1], axis=3)
    center_5=dense_block(center_4_3_2_1,[[1,256],[3,256],[1,1024]])
    center_5_4_3_2_1=concatenate([center_5, center_4_3_2_1], axis=3)
    center=dense_block(center_5_4_3_2_1,[[1,256],[3,256],[1,1024]])


    # center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    # center = BatchNormalization()(center)
    # center = Activation('relu')(center)
    # center = Conv2D(1024, (3, 3), padding='same')(center)
    # center = BatchNormalization()(center)
    # center = Activation('relu')(center)
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

    # 1024


   

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)

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
    # model=My_unet_512()

if __name__ == "__main__":
    main()

   