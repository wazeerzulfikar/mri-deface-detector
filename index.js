var http = require('http');
const tf = require('@tensorflow/tfjs');

const data = require('./public/data/mri_32.json')
const data_64 = require('./public/data/mri_64.json')

var path = require('path')

// var jimp = require('jimp')
// var sharp = require('sharp');
var pica = require('pica');


var nifti = require('nifti-js')
var niftijs = require('nifti-reader-js')
var pako = require('pako')
var ndarray = require('ndarray')
var math = require('mathjs')

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

	reader.onload = function(e) {

		if (e.target.readyState== FileReader.DONE) {
			if (!e.target.result) {
				return;
			} else {

				// METHOD 1 - Using NIFTI-Reader-JS

				// var file = e.target.result;
				// if (niftijs.isCompressed(file)) {
				// 	file = niftijs.decompress(file);
				// }

				// if (niftijs.isNIFTI(file)) {
				// 	var niftiHeader = niftijs.readHeader(file);
				// 	var image = niftijs.readImage(niftiHeader,file);
				// }

				// // Check if contents are correct using sum of pixels

				// var check = new Uint8Array(image)
				// console.log(check.reduce((a,b)=>a+b,0));


				//  METHOD 2 - Using NIFTI-JS

				var unzipped = pako.inflate(e.target.result);
				var contents = nifti.parse(unzipped);
				console.log(contents)

				// Check if contents are correct using sum of pixels

				var check = new Uint8Array(contents.data)
				// const uniqueValues = [...new Set(check)]; 
				console.log(check.reduce((a,b)=>a+b,0));


				// preprocess(contents);
			}
		}
	};

	reader.readAsArrayBuffer(file);

}

function preprocess(contents) {
	// Main function for preprocessing the contents of the NIFTI file, before feeding to model

	normalArray = Array.prototype.slice.call(contents.data);
	var mat = math.matrix(normalArray)
	var dimensions = contents.sizes.slice()
	console.log(dimensions)

	// Reshape takes too much time - Need to check
	// mat = math.reshape(mat,dimensions)

	var height = dimensions[0];
	var width = dimensions[1];
	var depth = dimensions[2];

	// Obtain mean across dimension
	var dim_0 = math.mean(mat,0);
	var dim_1 = math.mean(mat,1);
	var dim_2 = math.mean(mat,2);


}

function resize() {
	// Function for checking resizing of image.

	var dim_0_64 = math.matrix(data_64.dim_0)
	var dim_1_64 = math.matrix(data_64.dim_1)
	var dim_2_64 = math.matrix(data_64.dim_2)

	math.reshape(dim_0_64, [64,64])

	// pica's resize buffer apparently not a function. But its present in the documentation

	pica.resizeBuffer(dim_0_64.toArray(), 32, 32).then((image)=>console.log('Success')).catch((err)=>console.log(err))


	// jimp needs mime string in buffer

	// var dim_0_64_resized = jimp.read(Buffer.from(dim_0_64.toArray())).then(function (image) {
 //    // console.log('Sucess')
 	//	return image.resize(256, 256)
	// }).catch(function (err) {
	//     // handle an exception
	// });


	// Sharp not browserifying

	// sharp(Buffer.from(dim_0_64.toArray())
 //  	.(32, 32)


	// console.log(dim_0_64)
	var dim_0_32 = math.matrix(data.dim_0)

	// console.log(dim_0_64_resized.toArray().reduce((a,b)=>a+b,0));
	console.log(dim_0_32.toArray().reduce((a,b)=>a+b,0));

	
	
}


async function test(mri) {

	// Function to test the model with given mri model

	var dim_0_32 = math.matrix(mri.dim_0)
	var dim_1_32 = math.matrix(mri.dim_1)
	var dim_2_32 = math.matrix(mri.dim_2)

	dim_0_32 = math.multiply(dim_0_32, 1/255);
	dim_1_32 = math.multiply(dim_1_32, 1/255);
	dim_2_32 = math.multiply(dim_2_32, 1/255);

	var dim_0 = await tf.tensor4d(dim_0_32.toArray(), [1,32,32,1]);
	var dim_1 = await tf.tensor4d(dim_1_32.toArray(), [1,32,32,1]);
	var dim_2 = await tf.tensor4d(dim_2_32.toArray(), [1,32,32,1]);

	var dims = [dim_0, dim_1, dim_2];

	const prediction = model.predict(dims);
	const label = mri.label;
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

	test(data);


}

main();




