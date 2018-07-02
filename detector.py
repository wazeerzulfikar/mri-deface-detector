import cv2
import numpy as np
import os
import nibabel as nib
import random
import time

class Generator:

    def __init__(self, path, split=0.8):

        self.path = path
        self.mri_files = [i for i in os.listdir(path) if i.endswith('npz')]
        self.train_files, self.test_files = self.train_test_split(split=split)

            
    def train_test_split(self, split=0.8, shuffle=True):
        '''Shuffles, splits and returns train and test set as filenames'''
        
        if shuffle:
            random.shuffle(self.mri_files)

        split_index = int(split*len(self.mri_files))
        train_files = self.mri_files[:split_index]
        test_files = self.mri_files[split_index:]
        
        return train_files, test_files


    def load_npz(self,f):

        data = np.load(os.path.join(self.path,f))

        dim_0 = data['dim_0']
        dim_1 = data['dim_1']
        dim_2 = data['dim_2']
        label = data['label']

        return ([dim_0, dim_1, dim_2], label)


    def augment(self, images, target_size=None):
        '''Function for augmenting MRI images while training, to increase generalization'''

        sometimes = lambda aug : iaa.Sometimes(0.3,aug)

        # The commented augmentations sometimes destroyed the image. Need to discuss which ones are appropriate here

        seq = iaa.Sequential([
            sometimes(iaa.GaussianBlur(sigma=(0.0,2.0))),
            sometimes(iaa.ContrastNormalization((0.9,1.1))),
            sometimes(iaa.Multiply((0.95,1.05))),
          #  iaa.Sharpen(alpha=(0, 0.5), lightness=(0.9, 1.1)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),

            iaa.OneOf([
                iaa.Affine(rotate=(90)),
                iaa.Affine(rotate=(-90)),
                iaa.Affine(rotate=(0))
            ])
        ])

        dims = [list() for i in range(3)]

        if target_size:
                for i in range(len(images)):
                    img = images[i]
                    for j in range(len(dims)):
                        dims[j].append(cv2.resize(img[j].astype('float'), target_size))

        seq_det = seq.to_deterministic()

        aug_mri = []

        for i in range(3):
            aug_images = seq_det.augment_images(dims[i])
            aug_mri.append(np.expand_dims(aug_images, axis=3)/255.)

        return aug_mri


    def batch_read(self, batch_files, target_size):

        mri_images = list()
        labels = list()

        for e,f in enumerate(batch_files):
            mri, label = self.load_npz(f)
            mri_images.append(mri)
            labels.append(label)

        aug_mri = self.augment(mri_images, target_size)

        return (aug_mri, np.array(labels))



    def keras_generator(self, batch_size = 16, train=True):

        sizes = [(64,64)]

        while True:
            
            if train:
            
                random.shuffle(self.train_files)
                    
                for i in range(0, len(self.train_files), batch_size):
                    batch_files = self.train_files[i:i+batch_size]
                    batch_x, batch_y = self.batch_read(batch_files, target_size=random.choice(sizes))
                        
                    yield (batch_x , batch_y)

            else:
                random.shuffle(self.test_files)
                    
                for i in range(0, len(self.test_files), batch_size):
                    batch_files = self.test_files[i:i+batch_size]
                    batch_x, batch_y = self.batch_read(batch_files, target_size=random.choice(sizes))
                        
                    yield (batch_x, batch_y)


    def test_keras_generator(self, batch_size=4):

        import matplotlib.pyplot as plt

        g = self.keras_generator(batch_size=batch_size)
        batch_x, batch_y = next(g)

        plt.imshow(np.squeeze(batch_x[0][0]), 'gray')
        plt.show()

        plt.imshow(np.squeeze(batch_x[1][0]), 'gray')
        plt.show()

        plt.imshow(np.squeeze(batch_x[2][0]), 'gray')
        plt.show()


import imgaug as ia
from imgaug import augmenters as iaa
from keras import layers, models
from keras import regularizers
import keras.backend as K

# In[22]:
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


# In[23]:
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


# Test to check generator
generator = Generator('np_data')

n_train = len(generator.train_files)
n_test = len(generator.test_files)

print('Number of train images :', n_train)
print('Number of test images :', n_test)

generator.test_keras_generator(batch_size=4)

# Create necessary folders for logging and saving
#os.makedirs('models',exist_ok=True)
#os.makedirs('logs',exist_ok=True)

from keras.callbacks import ModelCheckpoint, CSVLogger

batch_size = 16

train_gen = generator.keras_generator(batch_size=16, train=True)
val_gen = generator.keras_generator(batch_size=16, train=True)

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