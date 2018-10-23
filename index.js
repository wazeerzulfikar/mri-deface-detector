var tf = require('@tensorflow/tfjs');

var path = require('path');
var Jimp = require('jimp');
var nj = require('numjs');

var detector = require('mri-deface-detector');

let model;

const messageElement = document.getElementById('message');
const statusElement = document.getElementById('status');
const imageElement = document.getElementById('images');
const inputElement = document.getElementById('file_input');

function clearElement(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

function displayImage(image) {
  image.getBase64(Jimp.MIME_JPEG, function(err, src) {
    const img = document.createElement('img');
    img.setAttribute('src', src);
    imageElement.appendChild(img);
  });
}

function showResults(result) {
  console.log('Prediction : ' + result);

  var status = `Prediction : ${result[0].toFixed(2)} `;

  statusElement.innerText = status;
  statusElement.innerText += '\n';

  if (result[0] < 0.5) {
    statusElement.innerText += ' It has NOT been defaced.';
  } else {
    statusElement.innerText += ' It has been defaced.';
  }
}

function readFile(e) {
  var file = e.target.files[0];

  var reader = new FileReader();

  reader.onerror = e => console.log('error');

  reader.onload = function(e) {
    if (e.target.readyState == FileReader.DONE) {
      if (!e.target.result) {
        return;
      } else {
        clearElement(imageElement);
        clearElement(statusElement);

        var file = e.target.result;

        console.log(file)

        var mri = detector.readNifti(file, error => {
          statusElement.innerText = error;
        });
        if (mri == undefined) return;

        // detector.detectDefaceCustom(model, mri, 'slice', 32, displayImage).then(showResults);
        detector.detectDeface(mri, displayImage).then(showResults);
      }
    }
  };

  reader.readAsArrayBuffer(file);
}

async function main() {
  model = await detector.loadModel(path.join('model_js', 'model.json'), model => {
    console.log('Model Has Been Loaded');
  });

  inputElement.addEventListener('change', readFile, false);
}

main();
