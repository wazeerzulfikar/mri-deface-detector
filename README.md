# MRI Deface Detector

[WIP] A deployable JS tool using Deep Learning to detect if defacing has been done on MRI Scans.

## How to Run the Detector on your Browser

Using npm:


```
npm install
npm run watch
```

Upload a `NIFTI` file to see results.

## Build your own Detector

The module has two components:

- Deep Learning Component (python)
- Deface Detector Tool (javascript)

## Deep Learning Component

### Data Handling

Dataset Structure : 

Data
+-- Undefaced
|	+-- image1.nii.gz 
|	+-- image2.nii.gz 
|	+-- ..
+-- Defaced
|	+-- image3.nii.gz 
|	+-- image4.nii.gz 
|	+-- ..

The 3 Dimensional MRI data is preprocessed to obtain 3 crossections using one of Mean or Slice. Examples for each of the two are given below.

### Mean

### Slice

For faster experimentation, the mri data is first cached as npz files.

To do this run :

`python load_dataset.py --load_path path/to/dataset1 path/to/dataset2 ..\
						--save_path path/to/save/cache
						--preprocess [Optional] mean/slice`

It is possible to keep appending more data as acquired to the cache by just running `load_dataset` using the same `save_path` again.

The existing model is trained using the IXI-Dataset and pydeface to create the corresponding dataset.

### Train the Model

To train the model :

`python detector.py --load_path path/to/npz/files`

Set the `--export_js` flag to True for automatic conversion of the best model to a TensorFlowJS usable form.
