const tf = require('@tensorflow/tfjs');

// Only for debugging purposes locally
// const data = require('./public/data/mri_full_uint8.json')
const data_64 = require('./public/data/mri_64.json')

var path = require('path')

var Jimp = require('jimp')

var nifti = require('nifti-js')
var niftijs = require('nifti-reader-js')
var pako = require('pako')
var nj = require('numjs')

let model;

const messageElement = document.getElementById('message');
const statusElement = document.getElementById('status');
const imageElement = document.getElementById('images');


async function loadModel(file, callback) {
	model = await tf.loadModel(path.join('model','model.json'));
	callback(model);
}

function clearElement(element){
	while (element.firstChild) {
    	element.removeChild(element.firstChild);
	}
}

function readFile(e) {

	file = e.target.files[0];

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

				var file = e.target.result;
				if (niftijs.isCompressed(file)) {
					file = niftijs.decompress(file);
					console.log('Decompressed')
				}

				if (niftijs.isNIFTI(file)) {
					var niftiHeader = niftijs.readHeader(file);
					var dimensions = niftiHeader.dims.slice(1,4).reverse();
					console.log('Dimensions : '+dimensions);
					var image = niftijs.readImage(niftiHeader,file);
				} else {
					statusElement.innerText = `Error! Please provide a valid NIFTI file.`;
					return;
				}

				var image = new Int16Array(image);

				// CheckSum
				// console.log(image.reduce((a,b)=>a+b,0));


				//  METHOD 2 - Using NIFTI-JS

				// var unzipped = pako.inflate(e.target.result);
				// var contents = nifti.parse(unzipped);
				// var image = contents.data

				// Check if contents are correct using sum of pixels

				// var check = new int16Array(contents.data)
				// const uniqueValues = [...new Set(check)]; 

				// var image = new Uint8Array(Array.prototype.slice.call(image))

				var slices = await preprocess(image, dimensions)

				test(...slices);
			}
		}
	};

	reader.readAsArrayBuffer(file);

}


async function preprocess(contents, dimensions) {
	// Main function for preprocessing the contents of the NIFTI file, before feeding to model

	var img = nj.float64(contents);

	img = nj.uint8(nj.multiply(nj.divide(img, img.max()-img.min()),255));

	img = img.reshape(dimensions);

	var slices = []

	var key = [null, null, null]

	for (var i=0;i<dimensions.length;i++) {
		var key = [null, null, null]
		key[i] = dimensions[i]/2;
		var slice = img.pick(...key)
		slices.push(slice.T)
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
			statusElement.innerText += ' It has been defaced.'
		} else{
			statusElement.innerText += ' It has NOT been defaced.'
		}
	});
	
}

async function main() {

	const inputElement = document.getElementById('file_input');
	inputElement.addEventListener('change', readFile, false);

	await loadModel('models/model.json', (model) => {
		console.log('Model Has Been Loaded');
		})

}

main();
