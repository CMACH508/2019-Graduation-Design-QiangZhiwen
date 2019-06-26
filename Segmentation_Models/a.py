from keras.models import *
import cv2
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import glob
import io
import skimage.io as io
import skimage.transform as trans
import numpy as np
import os

model = load_model('saved_model/medical_thesis_elas_less.hdf5')

def saveResult(name,save_path, npyfile, num_class=3):
    for i, item in enumerate(npyfile):
        name = name.split('/')[-1].split('.')[0]
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, "%s_predict.png" % name), img)


path = "327medical/test_1050/"
for i in glob.glob(path + '*.png'):
    img = cv2.imread(i)
#     a=np.zeros((512,512,3))
#     a[:,:,0]=img
#     a[:,:,1]=img
#     a[:,:,2]=img

#     img=a

    print(img.shape)
    
    img = img / 255
    img = np.reshape(img, (1,)+img.shape)
    result = model.predict(img)
    saveResult(i,path, result)