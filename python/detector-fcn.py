import cv2
import numpy as np
import os
import nibabel as nib
import random
import time
from keras import layers, models
from keras import regularizers
import keras.backend as K

from utils import Dataset, Generator

paths = ['../mri_data/IXI-Dataset/T1-Dataset', '../mri_data/IXI-Defaced/T1-Defaced']

dataset = Dataset(paths, 'np_data')
dataset.load_save_images()

generator = Generator('np_data')

n_train = len(generator.train_files)
n_test = len(generator.test_files)

print('Number of train images :', n_train)
print('Number of test images :', n_test)

# Test to check generator
# generator.test_keras_generator(batch_size=4)

def relu6(x):
    return K.relu(x, max_value=6)


def _Conv_BN_RELU(x, filters=32, kernel=3, strides=1, padding='same'):
    '''Helper to create a modular unit containing Convolution, BatchNormalizaton and Activation'''

    x = layers.Conv2D(filters,kernel,strides=strides,padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu6)(x)
    return x  


def create_submodel():
    '''The feature extracting submodel for which shares parameters'''

    inp = layers.Input(shape=(None,None,1))

    conv1 = _Conv_BN_RELU(inp, filters=8, kernel=3, strides=1, padding='same')
    conv1 = _Conv_BN_RELU(conv1, filters=8, kernel=3, strides=2, padding='same')

    conv2 = _Conv_BN_RELU(conv1, filters=16, kernel=3, strides=1, padding='same')
    conv2 = _Conv_BN_RELU(conv2, filters=16, kernel=2, strides=2, padding='same')
  #  conv2 = layers.MaxPooling2D()(conv2)

    conv3 = _Conv_BN_RELU(conv2, filters=24, kernel=3, strides=1, padding='same')
    conv3 = _Conv_BN_RELU(conv3, filters=24, kernel=2, strides=2, padding='same')
  #  conv3 = layers.MaxPooling2D()(conv3)

    conv4 = _Conv_BN_RELU(conv3, filters=32, kernel=3, strides=1, padding='same')
    conv4 = _Conv_BN_RELU(conv4, filters=64, kernel=2, strides=2, padding='same')

    out = layers.GlobalAveragePooling2D()(conv4)

    submodel = models.Model(inp,out)
    
    return submodel


def create_model():
    '''Assembles all the submodels into a unified single model'''

    inp1 = layers.Input(shape=(None,None,1), name='input_1')
    inp2 = layers.Input(shape=(None,None,1), name='input_2')
    inp3 = layers.Input(shape=(None,None,1), name='input_3')

    submodel = create_submodel()

    one = submodel(inp1)
    two = submodel(inp2)
    three = submodel(inp3)

    concat = layers.Concatenate()([one,two,three])
    dropout = layers.Dropout(0.2)(concat)
    out = layers.Dense(1,activation='sigmoid',name='output_node')(dropout)

    model = models.Model(inputs=[inp1,inp2,inp3],outputs=out)

    return model

# Defining customized metrics
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# Create and compile Keras model
model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',sensitivity, specificity])
print(model.summary())

# Create necessary folders for logging and saving
#os.makedirs('models',exist_ok=True)
#os.makedirs('logs',exist_ok=True)

from keras.callbacks import ModelCheckpoint, CSVLogger

#sizes = [(64,64), (128,128), (196,196), (224,224), (256,256)]
sizes = [(64,64)]

batch_size = 16

train_gen = generator.keras_generator(batch_size=16, train=True, augment=True, target_sizes=sizes)
val_gen = generator.keras_generator(batch_size=16, train=False, augment=False, target_sizes=sizes)

checkpoint = ModelCheckpoint(filepath='models/model_best.h5', save_best_only=True, monitor='val_loss',
                             save_weights_only=False)

csv_logger = CSVLogger('logs/training.log')

# Training with checkpoints for saving and logging results
model.fit_generator(train_gen, steps_per_epoch=n_train//batch_size,
                    validation_data=val_gen, validation_steps=n_test//batch_size,
                    epochs=30, callbacks=[checkpoint, csv_logger])


# Save and load model
model.save('models/model_final_v2.h5')

from keras.models import load_model
# model = load_model('models/model_best.h5')