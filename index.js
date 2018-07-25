var http = require('http');
const tf = require('@tensorflow/tfjs');

const data = require('./public/data/mri_32.json')
const data_64 = require('./public/data/mri_64.json')

var path = require('path')

// var jimp = require('jimp')
// var jimp = require('jimp');
// var pica = require('pica');


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

function readFile(evt) {

	file = evt.target.files[0];

	var reader = new FileReader();

	reader.onerror = (e) => console.log('error');

	reader.onload = function(e) {

		if (e.target.readyState== FileReader.DONE) {
			if (!e.target.result) {
				return;
			} else {
				var unzipped = pako.inflate(e.target.result);
				// var unzipped = niftijs.decompress(e.target.result);
				// console.log(unzipped)
				// var niftiHeader = niftijs.readHeader(unzipped);
				// var contents = niftijs.readImage(niftiHeader,unzipped);

				var contents = nifti.parse(unzipped);
				// normalArray = Array.prototype.slice.call(contents.data);
				console.log(contents)
				var check = new Uint8Array(contents.data)
				// const uniqueValues = [...new Set(check)]; 
				console.log(check.reduce((a,b)=>a+b,0));


				// console.log(contents.data.reduce((a,b)=>a+b,0))

				// console.log(mat);
				
				// preprocess(contents);

			}
		}
	};

	reader.readAsBinaryString(file);

}

function preprocess(contents) {

	normalArray = Array.prototype.slice.call(contents.data);
	var mat = math.matrix(normalArray)
	var dimensions = contents.sizes.slice()
	console.log(dimensions)
	// console.log(mat)

	// mat = math.reshape(mat,dimensions)

	// var height = dimensions[0];
	// var width = dimensions[1];
	// var depth = dimensions[2];

	// var dim_0 = math.mean(mat,0);
	// var dim_1 = math.mean(mat,1);
	// var dim_2 = math.mean(mat,2);

	var m = math.matrix(data.dim_0)
	console.log(m)

	m = math.reshape(m, [16,16,4])
	// m = math.resize(m,[24,24])
	console.log('hello')
	console.log(math.mean(m,0))
	console.log(math.mean(m,1))
	console.log(math.mean(m,2))

	// console.log(dim_1)
	// console.log(dim_2)



}

function readMRI() {
	var mri_data;
	

	var dim_0_64 = math.matrix(data_64.dim_0)
	var dim_1_64 = math.matrix(data_64.dim_1)
	var dim_2_64 = math.matrix(data_64.dim_2)

	math.reshape(dim_0_64, [64,64])

	// pica.resizeBuffer(dim_0_64.toArray(), 32, 32).then((image)=>console.log('Success')).catch((err)=>console.log(err))


	// console.log(dim_0_64)
	
	
}


async function test(mri) {

	var dim_0_32 = math.matrix(data.dim_0)
	var dim_1_32 = math.matrix(data.dim_1)
	var dim_2_32 = math.matrix(data.dim_2)

	dim_0_32 = math.multiply(dim_0_32, 1/255);
	dim_1_32 = math.multiply(dim_1_32, 1/255);
	dim_2_32 = math.multiply(dim_2_32, 1/255);

	var dim_0 = await tf.tensor4d(dim_0_32.toArray(), [1,32,32,1]);
	var dim_1 = await tf.tensor4d(dim_1_32.toArray(), [1,32,32,1]);
	var dim_2 = await tf.tensor4d(dim_2_32.toArray(), [1,32,32,1]);

	var dims = [dim_0, dim_1, dim_2];

	const prediction = model.predict(dims);
	statusElement.innerText = `Prediction : ${prediction}`;

}

async function main() {

	const inputElement = document.getElementById('file_input');
	inputElement.addEventListener('change', readFile, false);

	await loadModel('models/model.json', (model) => {
		messageElement.innerText = 'Model Has Been Loaded';
		})

	await readMRI();

	test(data);


}

main();




