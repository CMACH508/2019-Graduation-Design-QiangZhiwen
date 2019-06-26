# -*- coding: utf-8 -*-
"""
Created on 2019-3-12
@author: LeonShangguan
"""
from generater import *
from keras.models import *
import cv2
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

model = load_model('my_model.h5')

# testGene = testGenerator("aug/val/image/")
# results = model.predict_generator(testGene, 30, verbose=1)
# saveResult("aug/val/image", results)


img = io.imread("1.jpg", as_gray=False)
img = img / 255
# img_copy = io.imread("1.jpg", as_gray=True)/255
flag_multi_class = False
img = trans.resize(img, (256, 256))
print(img.shape)
img = np.reshape(img, img.shape) if (not flag_multi_class) else img
print(img.shape)
img = np.reshape(img, (1,)+img.shape)
print(img.shape)
result = model.predict(img)
print(result.shape)
saveResult("", result)
