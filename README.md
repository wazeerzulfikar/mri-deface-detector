# MRI Deface Detector

[WIP] A deployable JS tool using Deep Learning to detect if defacing has been done on MRI Scans.

## How to Run the Detector on your Browser

Using npm:

```
npm install
npm run watch
```

Upload a `NIFTI` file to see results.

Note : The existing model has been trained on the [IXI-Dataset](http://brain-development.org/ixi-dataset/), where [pydeface](https://github.com/poldracklab/pydeface) to create the corresponding defaced dataset.

## Build your own Detector

The module has two components:

- Deep Learning Model(python)
- Deface Detector Tool (javascript)

### Deep Learning Model

#### Dataset Preparation

Dataset Structure : 

```
Dataset
│
└───Undefaced
│   │	image1.nii.gz 
│   │	image2.nii.gz 
│   │	...
│
└───Defaced
    │	image3.nii.gz 
    │	image4.nii.gz
    │	...
```

The three-dimensional MRI scan is preprocessed to obtain 3 two-dimensional cross-sections using one of mean or slice methods. Examples for each of the two are given below.

##### Mean
Arithmetic mean along each of the dimensions

![mean](assets/undefaced_mean.jpg)

##### Slice
Center slice along each of the dimensions

![slice](assets/undefaced_slice.jpg)

Note : The existing model uses mean preprocessing.

For faster experimentation, the mri data is first cached as npz files.

To do this run :

```
python load_dataset.py --load_path path/to/dataset1 path/to/dataset2 .. \
		       --save_path path/to/save/cache \
		       --preprocess [Optional] mean/slice
```

It is possible to keep appending more data as acquired to the cache by just running `load_dataset` using the same `save_path` again.

#### Train the Model

To train the model :

```
python detector.py --load_path path/to/npz/files \
		   --export_js False
```

Set the `--export_js` flag to True for automatic conversion of the best model to a TensorFlowJS usable form.


### Deface Detector Tool

#### Port the Custom Model to Deface Detector

- The TensorFlowJS model consists of the model structure in the form a JSON file and the weights as shards.
- Copy all the components into the `/public` folder
- Kick Start the detector!
