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
const statusElement = document.getElementById('status')

// statusElement.innerText = data.dim_0.length -1

async function loadModel(file, callback) {
	model = await tf.loadModel(path.join('model','model.json'));
	callback(model);
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

				// METHOD 1 - Using NIFTI-Reader-JS

				var file = e.target.result;
				if (niftijs.isCompressed(file)) {
					file = niftijs.decompress(file);
					console.log('decompress')
				}

				if (niftijs.isNIFTI(file)) {
					var niftiHeader = niftijs.readHeader(file);
					console.log(niftiHeader.dims)
					var image = niftijs.readImage(niftiHeader,file);
				}

				// // Check if contents are correct using sum of pixels
				// console.log(new DataView(image))

				var image = new Int16Array(image)
				console.log(image)

				console.log(image.reduce((a,b)=>a+b,0));


				//  METHOD 2 - Using NIFTI-JS

				// var unzipped = pako.inflate(e.target.result);
				// var contents = nifti.parse(unzipped);
				// var image = contents.data
				// console.log(contents)

				// Check if contents are correct using sum of pixels

				// var check = new int16Array(contents.data)
				// const uniqueValues = [...new Set(check)]; 
				// console.log(check);
				// console.log(image)


				// var image = new Uint8Array(Array.prototype.slice.call(image))
				// console.log(image)
				// console.log(data.image)
				// console.log(image.reduce((a,b)=>a+b,0));
				// console.log(data.image.reduce((a,b)=>a+b,0));


				var dims = await preprocess(image)
				var dim_0 = dims[0]
				var dim_1 = dims[1]
				var dim_2 = dims[2]

				test(dim_0, dim_1, dim_2,1);
			}
		}
	};

	reader.readAsArrayBuffer(file);

}


async function preprocess(contents) {
	// Main function for preprocessing the contents of the NIFTI file, before feeding to model

	var njarray = nj.uint8(contents);

	var dimensions = [256, 256, 150];

	njarray = njarray.reshape(dimensions);

	console.log('Reshape Done')


	var dims = []

	var key = [null, null, null]

	// for (var i=0;i<dimensions.length;i++) {
	// 	var key = [null, null, null]
	// 	key[i] = dim/2;
	// 	var dim = njarray.pick(key)
	// 	console.log(dim);
	// 	dims.push(dim)
	// }
	var dim = njarray.pick(128,null,null)

	// console.log(dim);
	// console.log(dim.flatten());

	dims.push(dim)
	var dim = njarray.pick(null,128,null)

	// console.log(dim);
	dims.push(dim)
	var dim = njarray.pick(null,null,75)

	// console.log(dim);
	dims.push(dim)


	return dims

}


function resize(img_data, target_height, target_width) {
	// Function for  resizing of image.

	// var image_data = img.selection.data;

	var width = img_data.shape[0]
	var height = img_data.shape[1]
	img_data = img_data.flatten().tolist();

	console.log(height)
	console.log(width)
	console.log('Before copying')

	var cross = new Jimp(height,width, function (err, image) {

		let buffer = image.bitmap.data
		var i = 0;
		for(var x=0; x<height*width*4;x+=4) {
			// const offset = (y*width+x)*4;
			buffer[x] = img_data[i];
			buffer[x+1] = img_data[i];
			buffer[x+2] = img_data[i];
			buffer[x+3] = 255;
			i++;
			
		}

		image.getBase64(Jimp.MIME_JPEG, function (err, src) {
        const img = document.createElement('img');
        img.setAttribute('src', src);
        document.body.appendChild(img)});

        image.resize(target_height,target_width);	
	});

	var i = 0;
	let resized_image_data = new Float64Array(target_height*target_width);
    for(var x=0; x<height*width*4;x+=4) {
		resized_image_data[i] = cross.bitmap.data[x];
		i++;
		
	}
		console.log(resized_image_data.reduce((a,b)=>a+b,0));


    return resized_image_data;
	
	// const uniqueValues = [...new Set(cross.bitmap.data)]; 

	// console.log(uniqueValues)
	
	
}


async function test(dim_0, dim_1, dim_2, label) {

	// Function to test the model with given mri model


	var dim_0_32 = nj.float64(resize(dim_0, 32, 32));
	var dim_1_32 = nj.float64(resize(dim_1, 32, 32));
	var dim_2_32 = nj.float64(resize(dim_2, 32, 32));

	dim_0_32 = dim_0_32.divide(255);
	dim_1_32 = dim_1_32.divide(255);
	dim_2_32 = dim_2_32.divide(255);

	console.log(dim_0_32.flatten().tolist().reduce((a,b)=>a+b,0));
	console.log(dim_1_32.flatten().tolist().reduce((a,b)=>a+b,0));
	console.log(dim_2_32.flatten().tolist().reduce((a,b)=>a+b,0));



	var dim_0 = await tf.tensor4d(dim_0_32.flatten().tolist(), [1,32,32,1]);
	var dim_1 = await tf.tensor4d(dim_1_32.flatten().tolist(), [1,32,32,1]);
	var dim_2 = await tf.tensor4d(dim_2_32.flatten().tolist(), [1,32,32,1]);

	var dims = [dim_0, dim_1, dim_2];

	var prediction = model.predict(dims);
	console.log(prediction)
	statusElement.innerText = `Prediction : ${prediction} , Actual : ${label}`;

}

async function main() {

	const inputElement = document.getElementById('file_input');
	inputElement.addEventListener('change', readFile, false);

	await loadModel('models/model.json', (model) => {
		messageElement.innerText = 'Model Has Been Loaded';
		})

	// await resize();

	// var dims = await preprocess(data.image)
	// var dim_0 = dims[0]
	// var dim_1 = dims[1]
	// var dim_2 = dims[2]

	// test(dim_0, dim_1, dim_2, data.label);

}

main();
