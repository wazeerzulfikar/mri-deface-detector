// var http = require('http');
var tf = require('@tensorflow/tfjs');

var data = require('./public/data/mri.json');
var path = require('path');
// const full_data = require('./public/mri_full.json')

var nifi = require('nifti-js');
var buffer = require('buffer');
console.log(data.dim_0.length);

let model;

async function loadModel(file) {
  model = await tf
    .loadModel(path.join(__dirname, 'public/model/model.json'))
    .then(function() {
      console.log('Model Has Been Loaded');
      test(data);
    })
    .catch(err => console.log(err));
}

async function readMRI(filename) {
  fs.readFile(filename, 'utf-8', function(err, data) {
    console.log('MRI readed');
    console.log(data.length);
  });
}

function test(mri) {
  var dim_0 = tf.tensor4d(data.dim_0, [1, 32, 32, 1]);
  var dim_1 = tf.tensor4d(data.dim_1, [1, 32, 32, 1]);
  var dim_2 = tf.tensor4d(data.dim_2, [1, 32, 32, 1]);

  var dims = [dim_0, dim_1, dim_2];

  const prediction = model.predict(dims);
  console.log(`Prediction : ${prediction}`);
}

function main() {
  loadModel('models/model.json');

  // await readMRI('./');

  // test(data);
}

main();
