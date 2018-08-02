import cv2
import numpy as np
import os
import nibabel as nib
import random
import time
import imgaug as ia 
from imgaug import augmenters as iaa

class Dataset:
    '''Class that handles loading the MRI data and saving it in a npz format for generator'''

    def __init__(self, paths, save_path, batch_size=100, verbose=1):
        
        self.paths = paths 
        self.save_path = save_path
        self.mri_files = list()
        self.verbose = verbose
        self.batch_size = batch_size

        for path in paths:
            if not os.path.exists(path):
                raise Exception('"{}" does not exist!'.format(path))
            self._load_files(path)

        print('Number of NIFTI files found : ', len(self.mri_files))


    def _load_files(self, path):
        '''Recursively goes into the path to extract all NIFTI image filenames (relative to path).
        Defaced and Undefaced MRI images are treated alike.'''

        if path.endswith('nii.gz'):
            self.mri_files.append(path)
        elif os.path.isdir(path):
            for file in os.listdir(path):
                self._load_files(os.path.join(path,file))


    def read_mri_image(self, filename):
        '''Utility function to read a single MRI image,
        as a numpy array'''
    
        img = nib.load(filename)
        return img.get_data()


    def minmax(self, img):
        '''MinMax normalization of image'''
        return (img/(np.max(img)-np.min(img)))*255

    def _batch_read(self, files, start_index, preprocess=None):
        '''Reads MRI images batch by batch. All defaced images are identified by a "deface" in the file name.'''

        for f in files:
            img = self.read_mri_image(f)
            img = self.minmax(img)

            if preprocess not in ['mean','slice']:
                 raise Exception(' Preprocess has to be one of [mean, slice]!')

            if preprocess=='mean':            
                dim_0 = self.minmax(np.mean(img,axis=0))
                dim_1 = self.minmax(np.mean(img,axis=1))
                dim_2 = self.minmax(np.mean(img,axis=2))

            elif preprocess=='slice':
                dimensions = img.shape
                dim_0 = img[dimensions[0]//2,:,:]
                dim_1 = img[:,dimensions[1]//2,:]
                dim_2 = img[:,:,dimensions[2]//2,]
            
            if 'deface' in f:
                label = 1
            else:
                label = 0

            self.save_as_npz(f, dim_0, dim_1, dim_2, label)


    def load_save_images(self):

        start = time.time()

        if self.verbose==1:
            import tqdm
            looper = tqdm.trange(0, len(self.mri_files),self.batch_size)
        else:
            looper = range(0,len(self.mri_files),self.batch_size)
        
        for i in looper:
            if self.verbose == 1:
                os.system('echo "Process Starting.."')
            if i+self.batch_size > len(self.mri_files):
                self._batch_read(self.mri_files[i:], i, preprocess='slice')
                if self.verbose == 1:
                    print(len(self.mri_files), " Done")
            else:
                self._batch_read(self.mri_files[i:i+self.batch_size], i, preprocess='slice')
                if self.verbose == 1:
                    print(i+self.batch_size, " Done")
                    
        print("Time Taken : ", time.time()-start)
        

    def save_as_npz(self, f, dim_0, dim_1, dim_2, label):

        f = f.split('/')[-1].replace('.nii.gz','.npz')
        savename = os.path.join(self.save_path, f)
        np.savez(savename, dim_0=dim_0, dim_1=dim_1, dim_2=dim_2, label=np.array(label))



class Generator:
    '''Class that handles split of MRI data, augmentations and supplying a generator for the
    keras model to train'''

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


    def preprocess(self, images, target_size=None, augment=True):
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

        dimensions = [list() for i in range(3)]

        if target_size:
            for i in range(len(images)):
                img = images[i]
                for dim in range(len(dimensions)):
                    dimensions[dim].append(cv2.resize(img[dim].astype('float'), target_size))

        seq_det = seq.to_deterministic()
        preprocessed_mri = []

        for dim in range(len(dimensions)):
            if augment:
                aug_images = seq_det.augment_images(dimensions[dim])
            else:
                aug_images = np.array(dimensions[dim])
            preprocessed_mri.append(np.expand_dims(aug_images, axis=3)/255.)

        return preprocessed_mri


    def batch_read(self, batch_files, target_size=None, augment=True):

        mri_images = list()
        labels = list()

        for e,f in enumerate(batch_files):
            mri, label = self.load_npz(f)
            mri_images.append(mri)
            labels.append(label)

        batch = self.preprocess(mri_images, target_size=target_size, augment=augment)

        return (batch, np.array(labels))



    def keras_generator(self, batch_size = 16, train=True, augment=True, target_size=[(64,64)]):

        while True:
            if train:
                random.shuffle(self.train_files)   

                for i in range(0, len(self.train_files), batch_size):
                    batch_files = self.train_files[i:i+batch_size]
                    batch_x, batch_y = self.batch_read(batch_files, target_size=random.choice(target_size), augment=augment)
                        
                    yield (batch_x , batch_y)

            else:
                random.shuffle(self.test_files)

                for i in range(0, len(self.test_files), batch_size):
                    batch_files = self.test_files[i:i+batch_size]
                    batch_x, batch_y = self.batch_read(batch_files, target_size=random.choice(target_size), augment=augment)
                        
                    yield (batch_x, batch_y)


    def test_keras_generator(self, batch_size=4):

        import matplotlib.pyplot as plt

        g = self.keras_generator(batch_size=batch_size)
        batch_x, batch_y = next(g)

        for i in range(3):
            plt.imshow(np.squeeze(batch_x[i][0]), 'gray')
            plt.show()

        