const tf = require('@tensorflow/tfjs');

var path = require('path')
var Jimp = require('jimp')
var niftijs = require('nifti-reader-js')
var pako = require('pako')
var nj = require('numjs')

let model;

const messageElement = document.getElementById('message');
const statusElement = document.getElementById('status');
const imageElement = document.getElementById('images');
const inputElement = document.getElementById('file_input');

async function loadModel(filename, callback) {
	model = await tf.loadModel(filename);
	callback(model);
}


function clearElement(element){
	while (element.firstChild) {
    	element.removeChild(element.firstChild);
	}
}


function readNIFTI(file) {

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
		statusElement.innerText = `Error! Please provide a valid NIFTI file.`;
		return;
	}

	return {
		image: new Int16Array(image),
		dimensions : dimensions
	};
}


function readFile(e) {

	var file = e.target.files[0];

	var reader = new FileReader();

	reader.onerror = (e) => console.log('error');

	reader.onload = async function(e) {

		if (e.target.readyState== FileReader.DONE) {
			if (!e.target.result) {
				return;
			}  
			else {

				clearElement(imageElement);
				clearElement(statusElement);

				// METHOD 1 - Using NIFTI-Reader-JS

				var filename = e.target.result;

				var contents = readNIFTI(filename);
				if (contents==undefined) return;

				var image = contents.image;
				var dimensions = contents.dimensions;

				var slices = await preprocess(image, dimensions, 'slice');

				test(...slices);
			}
		}
	};

	reader.readAsArrayBuffer(file);
}


function axisMean(img, dimensions, axis) {
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
}


function minmaxNormalize(img) {
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


async function preprocess(contents, dimensions, preprocess_method) {
	// Main function for preprocessing the contents of the NIFTI file, before feeding to model

	var img = nj.float64(contents);

	img = minmaxNormalize(img);

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
	
	return slices
}


function resize(img_data, target_height, target_width) {
	// Function for  resizing of image.

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

		image.getBase64(Jimp.MIME_JPEG, function (err, src) {
        const img = document.createElement('img');
        img.setAttribute('src', src);
        imageElement.appendChild(img)});

        image.resize(target_height,target_width);	
	});

	var i = 0;
	let resized_image_data = new Float64Array(target_height*target_width);
    for(var x=0; x<height*width*4;x+=4) {
		resized_image_data[i] = resized.bitmap.data[x];
		i++;
		
	}

    return resized_image_data;
}


async function test(slice_0, slice_1, slice_2, label) {
	// Function to test the model with given mri model

	var slice_0 = nj.float64(resize(slice_0, 32, 32));
	var slice_1 = nj.float64(resize(slice_1, 32, 32));
	var slice_2 = nj.float64(resize(slice_2, 32, 32));

	slice_0 = nj.divide(slice_0,255);
	slice_1 = nj.divide(slice_1,255);
	slice_2 = nj.divide(slice_2,255);

	var slice_0 = await tf.tensor4d(slice_0.flatten().tolist(), [1,32,32,1]);
	var slice_1 = await tf.tensor4d(slice_1.flatten().tolist(), [1,32,32,1]);
	var slice_2 = await tf.tensor4d(slice_2.flatten().tolist(), [1,32,32,1]);

	var slices = [slice_0, slice_1, slice_2];

	var prediction = model.predict(slices);
	prediction.data().then(function(result){

		console.log('Prediction : '+result);

		var status = `Prediction : ${result[0].toFixed(2)} `;
		if (label!=undefined){
			status += `Actual : ${label}`;
		}
		statusElement.innerText = status;
		statusElement.innerText +='\n'

		if(result[0]<0.5){
			statusElement.innerText += ' It has NOT been defaced.'
		} else{
			statusElement.innerText += ' It has been defaced.'
		}
	});
	
}


async function main() {

	inputElement.addEventListener('change', readFile, false);

	await loadModel(path.join('model_js','model.json'), (model) => {
		console.log('Model Has Been Loaded');
		})

}

main();
