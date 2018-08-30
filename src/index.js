const tf = require('@tensorflow/tfjs');

var path = require('path')
var Jimp = require('jimp')
var nj = require('numjs')

var readNifti = require('./readNifti')
var preprocess = require('./preprocess')
var predict = require('./predict')

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

function displayImage(image) {
	image.getBase64(Jimp.MIME_JPEG, function (err, src) {
		const img = document.createElement('img');
		img.setAttribute('src', src);
		imageElement.appendChild(img)
	});
}

function showResults(result) {
	console.log('Prediction : '+result);

	var status = `Prediction : ${result[0].toFixed(2)} `;

	statusElement.innerText = status;
	statusElement.innerText +='\n'

	if(result[0]<0.5){
		statusElement.innerText += ' It has NOT been defaced.'
	} else{
		statusElement.innerText += ' It has been defaced.'
	}
}

function readFile(e) {

	var file = e.target.files[0];

	var reader = new FileReader();

	reader.onerror = (e) => console.log('error');

	reader.onload = function(e) {

		if (e.target.readyState == FileReader.DONE) {
			if (!e.target.result) {
				return;
			}  
			else {

				clearElement(imageElement);
				clearElement(statusElement);

				var file = e.target.result;

				var contents = readNifti(file, (error)=> {statusElement.innerText = error;});
				if (contents==undefined) return;

				var image = contents.image;
				var dimensions = contents.dimensions;

				var slices = preprocess(image, dimensions, 'slice', displayImage);

				predict(model, ...slices).then(showResults);
			}
		}
	};

	reader.readAsArrayBuffer(file);
}


async function main() {
	
	await loadModel(path.join('model_js','model.json'), (model) => {
	console.log('Model Has Been Loaded');
	})

	inputElement.addEventListener('change', readFile, false);
}

main();
