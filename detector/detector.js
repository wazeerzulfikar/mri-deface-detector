var tf = require('@tensorflow/tfjs');
var utils = require('./utils');

var detector = {

  defaceDetect : async function (mri, callback) {
      var model_path = 'https://raw.githubusercontent.com/wazeerzulfikar/Deface-Detector/master/dist/model_js/model.json';
      var model = await utils.loadModel(model_path, model => console.log('Model Loaded'));
      var input_size = 32;
      var preprocess_method = 'slice';

      return detector.customPredict(model, mri, preprocess_method, input_size, callback);
  },

  customPredict : async function(
    model,
    mri,
    preprocess_method,
    input_size,
    callback
  ) {
    var image = mri.image;
    var dimensions = mri.dimensions;

    var slices = utils.preprocess(
      image,
      dimensions,
      input_size,
      preprocess_method,
      callback
    );

    var tensor_dims = [1, input_size, input_size, 1];

    var slice_0 = await tf.tensor4d(slices[0].flatten().tolist(), tensor_dims);
    var slice_1 = await tf.tensor4d(slices[1].flatten().tolist(), tensor_dims);
    var slice_2 = await tf.tensor4d(slices[2].flatten().tolist(), tensor_dims);

    var slices = [slice_0, slice_1, slice_2];

    var prediction = model.predict(slices);

    return prediction.data();
  },



}

module.exports = detector