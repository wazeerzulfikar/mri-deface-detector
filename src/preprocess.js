var nj = require('numjs')
var resize = require('./resize')

function axisMean(img, dimensions, axis) {
	// Arithmetic Mean along specified axis

	axes = [0,1,2];
	axes.splice(axis, 1);

	var slice = [];

	for(var i=0;i<dimensions[axes[0]];i++){

		for(var j=0;j<dimensions[axes[1]];j++) {
			key = [null,null,null];
			key[axes[0]]=i;
			key[axes[1]]=j;
			slice.push(img.pick(...key).mean())
		}
	}

	return nj.float64(slice).reshape([dimensions[axes[0]],dimensions[axes[1]]]);
}


function minmaxNormalize(img) {
	// MinMax Normalization to 0-255 scale

	var max_val = img.max();
	var min_val = img.min();
	img = nj.divide(img, max_val-min_val);
	img = nj.multiply(img,255);
	if(min_val<0) {
		img = nj.add(img,127.5);
	}

	return img
}


module.exports =  function (contents, dimensions, preprocess_method, callback) {
	// Main function for preprocessing the contents of the NIFTI file, before feeding to model

	var img = nj.float64(contents);

	img = minmaxNormalize(img);

	img = nj.uint8(img.reshape(dimensions));

	var slices = [];

	if (preprocess_method=='slice') {
		for (var i=0;i<dimensions.length;i++) {
			var key = [null, null, null];
			key[i] = dimensions[i]/2;
			var slice = img.pick(...key);
			slices.push(slice.T);
		}
	} 
	else if (preprocess_method=='mean') {
		for (var i=0;i<dimensions.length;i++) {
			slice = axisMean(img, dimensions, i).T;
			slices.push(minmaxNormalize(slice));
		}	

	}

	for(var i=0;i<dimensions.length;i++) {
		slices[i] = nj.float64(resize(slices[i], 32, 32, callback));
		slices[i] = nj.divide(slices[i], 255);
	}
	
	return slices
}
