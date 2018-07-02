import cv2
import numpy as np
import os
import nibabel as nib
import random
import time

class Dataset:

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

    def _batch_read(self, files, start_index):
        '''Reads MRI images batch by batch. All defaced images are identified by a "deface" in the file name.'''

        for f in files:
            img = self.read_mri_image(f)
            
            dim_0 = np.mean(img,axis=0)
            dim_1 = np.mean(img,axis=1)
            dim_2 = np.mean(img,axis=2)
            
            if 'deface' in f:
                label = 1
            else:
                label = 0

            self.save_as_npz(f, dim_0, dim_1, dim_2, label)

    def save_images(self):

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
                self._batch_read(self.mri_files[i:], i)
                if self.verbose == 1:
                    print(len(self.mri_files), " Done")
            else:
                self._batch_read(self.mri_files[i:i+self.batch_size], i)
                if self.verbose == 1:
                    print(i+self.batch_size, " Done")
                    
        print("Time Taken : ", time.time()-start)
        

    def save_as_npz(self, f, dim_0, dim_1, dim_2, label):

        f = f.split('/')[-1].replace('.nii.gz','.npz')
        
        savename = os.path.join(self.save_path, f)

        np.savez(savename, dim_0=dim_0, dim_1=dim_1, dim_2=dim_2, label=np.array(label))


if __name__ == '__main__':

    paths = ['../mri_data/IXI-Dataset/T1-Dataset', '../mri_data/IXI-Defaced/T1-Defaced']

    dataset = Dataset(paths, 'np_data')
    dataset.save_images()
