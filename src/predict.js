const tf = require('@tensorflow/tfjs');
var utils = require('./utils')

/**
* Predict
*
* Takes model, mri scan, preprocess method, input_size and a callback for displaying image
* and returns a promise which when resolved contains the
* floating point prediction of the model.
*/

module.exports = async function (model, mri, preprocess_method, input_size, callback) {

	var image = mri.image;
	var dimensions = mri.dimensions;

	var slices = utils.preprocess(image, dimensions, input_size, preprocess_method, callback);

	var tensor_dims = [1,input_size,input_size,1];

	var slice_0 = await tf.tensor4d(slices[0].flatten().tolist(), tensor_dims);
	var slice_1 = await tf.tensor4d(slices[1].flatten().tolist(), tensor_dims);
	var slice_2 = await tf.tensor4d(slices[2].flatten().tolist(), tensor_dims);

	var slices = [slice_0, slice_1, slice_2];

	var prediction = model.predict(slices);

	return prediction.data();
}
