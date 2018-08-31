var nj = require('numjs')
var Jimp = require('jimp')
var niftijs = require('nifti-reader-js')
const tf = require('@tensorflow/tfjs');

var utils = {

	/**
	* loadModel
	*
	* Given the filename, the function asynchronously loads
	* the model and the callback handles the response
	*/

	loadModel : async function (filename, callback) {
		model = await tf.loadModel(filename);
		callback(model);
		return model;
	},

	/**
	* readNifti
	*
	* The function takes a file, checks for NIFTI, reads it
	* and returns the necessary contents. The callback handles errors.
	*/

	readNifti : function (file, callback) {
			if (niftijs.isCompressed(file)) {
				var file = niftijs.decompress(file);
				console.log('Decompressed')
			}

			if (niftijs.isNIFTI(file)) {
				var niftiHeader = niftijs.readHeader(file);
				var dimensions = niftiHeader.dims.slice(1,4).reverse();
				console.log('Dimensions : '+dimensions);
				var image = niftijs.readImage(niftiHeader, file);
			} else {
				callback(`Error! Please provide a valid NIFTI file.`);
				return;
			}

			return {
				image: new Int16Array(image),
				dimensions : dimensions
			};
	},

	/**
	* preprocess
	*
	* Takes image and dimensions as read using readNifti, input_size indicating
	* the input shape to the trained model. Preprocess method is one of `slice/mean`.
	* Callback for displaying the image in the browser.
	*/

	preprocess : function (contents, dimensions, input_size, preprocess_method, callback) {
		// Main function for preprocessing the contents of the NIFTI file, before feeding to model

		var img = nj.float64(contents);

		img = utils.minmaxNormalize(img);

		img = nj.uint8(img.reshape(dimensions));

		var slices = [];

		if (preprocess_method=='slice') {
			for (var i=0;i<dimensions.length;i++) {
				var key = [null, null, null];
				key[i] = dimensions[i]/2;
				var slice = img.pick(...key);
				slices.push(slice.T);
			}
		} 
		else if (preprocess_method=='mean') {
			for (var i=0;i<dimensions.length;i++) {
				slice = axisMean(img, dimensions, i).T;
				slices.push(minmaxNormalize(slice));
			}	

		}

		for(var i=0;i<dimensions.length;i++) {
			slices[i] = nj.float64(utils.resizeImage(slices[i], input_size, callback));
			slices[i] = nj.divide(slices[i], 255);
		}
		
		return slices
	},

	/**
	* resizeImage
	*
	* Takes a grayscale image as a numjs NdArray, and target size. 
	* Callback to display the image on the browser.
	* Returns the resized image as FloatArray.
	*/

	resizeImage : function (img_data, target_size, callback) {
		// function for  resizing of image.

		console.log('Resizing..')

		var width = img_data.shape[0]
		var height = img_data.shape[1]
		img_data = img_data.flatten().tolist();

		var resized = new Jimp(height,width, function (err, image) {

			let buffer = image.bitmap.data
			var i = 0;
			for(var x=0; x<height*width*4;x+=4) {
				buffer[x] = img_data[i];
				buffer[x+1] = img_data[i];
				buffer[x+2] = img_data[i];
				buffer[x+3] = 255;
				i++;
			}

			callback(image);

	        image.resize(target_size,target_size);	
		});

		var i = 0;
		let resized_image_data = new Float64Array(target_size*target_size);
	    for(var x=0; x<height*width*4;x+=4) {
			resized_image_data[i] = resized.bitmap.data[x];
			i++;
			
		}

	    return resized_image_data;
	},

	/**
	* axisMean
	*
	* Preprocess method which takes 3D mri scan as numjs NdArray and 
	* returns the three slices after performing arithmetic mean preprocess.
	* More information on the readme.
	*/

	axisMean : function (img, dimensions, axis) {
		// Arithmetic Mean along specified axis

		axes = [0,1,2];
		axes.splice(axis, 1);

		var slice = [];

		for(var i=0;i<dimensions[axes[0]];i++){

			for(var j=0;j<dimensions[axes[1]];j++) {
				key = [null,null,null];
				key[axes[0]]=i;
				key[axes[1]]=j;
				slice.push(img.pick(...key).mean())
			}
		}

		return nj.float64(slice).reshape([dimensions[axes[0]],dimensions[axes[1]]]);
	},

	/**
	* minmaxNormalize
	*
	* Takes the image, and does the min max normalization on it. Important for 
	* successful nifti read.
	* Returns the normalized image.
	*/

	minmaxNormalize : function (img) {
		// MinMax Normalization to 0-255 scale

		var max_val = img.max();
		var min_val = img.min();
		img = nj.divide(img, max_val-min_val);
		img = nj.multiply(img,255);
		if(min_val<0) {
			img = nj.add(img,127.5);
		}

		return img
	}

}

module.exports = utils;
