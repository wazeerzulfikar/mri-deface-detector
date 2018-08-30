const tf = require('@tensorflow/tfjs');

/**
* Predict
*
* Takes the three slices of MRI to be fed to the model
* and returns a promise which when resolved contains the
* floating point prediction of the model.
*/

module.exports = async function (model, slice_0, slice_1, slice_2) {

	var slice_size = Math.sqrt(slice_0.shape[0]);

	var slice_0 = await tf.tensor4d(slice_0.flatten().tolist(), [1,slice_size,slice_size,1]);
	var slice_1 = await tf.tensor4d(slice_1.flatten().tolist(), [1,slice_size,slice_size,1]);
	var slice_2 = await tf.tensor4d(slice_2.flatten().tolist(), [1,slice_size,slice_size,1]);

	var slices = [slice_0, slice_1, slice_2];

	var prediction = model.predict(slices);

	return prediction.data();
}