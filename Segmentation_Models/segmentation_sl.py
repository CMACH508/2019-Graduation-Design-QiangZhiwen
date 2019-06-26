# -*- coding: utf-8 -*-
"""
Created on 2019-3-11
@author: LeonShangguan
"""
from keras.utils import multi_gpu_model
from atmp import *
import numpy,cv2
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.callbacks import ModelCheckpoint
from generater import *

BACKBONE = 'seresnet50'
preprocess_input = get_preprocessing(BACKBONE)

# load your data
# (x_train, y_train), (x_val, y_val) = mnist.load_data()

# preprocess input
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

# define model


model = Unet(BACKBONE, encoder_weights='imagenet')
model.summary()

model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

# fit model
# model.fit(
#     x=x_train,
#     y=y_train,
#     batch_size=16,
#     epochs=10,
#     validation_data=(x_val, y_val),
# )


# data_gen_args = dict(fill_mode='nearest')
# myGene = trainGenerator(2, 'aug/train', 'image', 'label', data_gen_args, save_to_dir=None)
# model.fit_generator(myGene, steps_per_epoch=140, epochs=40, callbacks=[model_checkpoint])




x_train,y_train=generate_data_format("327medical/elas/image/","327medical/elas/label/")



print(x_train.shape)
print(y_train.shape)


callbacks = [EarlyStopping(monitor='loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='loss',
                             filepath='saved_model/medical_thesis_elas_less.hdf5',
                             save_best_only=True),
             TensorBoard(log_dir='logs')]

history=model.fit(x_train, y_train,
          batch_size=8,
          epochs=500,
          verbose=1,
          callbacks=callbacks)

# model.fit_generator(datagen.flow(x_train, y_train, batch_size=5),
#                     steps_per_epoch=len(x_train) / 5, epochs=500,
#                     verbose=1,callbacks=callbacks)






# model.fit_generator(datagen.flow(x_train, y_train, batch_size=5),
#                     steps_per_epoch=15, epochs=1000,
#                     verbose=1,callbacks=callbacks)

# history = model.fit_generator((x_train, y_train, batch_size=batch_size), epochs=epochs,
#                              validation_data=(x_val, y_val),
#                               steps_per_epoch=600,verbose=1,callbacks=[model_checkpoint])



# model.fit_generator(generator=train_generator(),
#                     steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
#                     epochs=epochs,
#                     verbose=2,
#                     callbacks=callbacks,
#                     validation_data=valid_generator(),
#                     validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
