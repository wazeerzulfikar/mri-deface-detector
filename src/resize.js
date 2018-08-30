var Jimp = require('jimp')

/**
 * Resize an image which is stored as a numjs vector
 *
 * @param {NdArray} img_data
 * @param {int} target_height
 * @param {int} target_width
 * @return {FloatArray}
 */

module.exports = function (img_data, target_height, target_width, callback) {
	// Function for  resizing of image.

	console.log('Resizing..')

	var width = img_data.shape[0]
	var height = img_data.shape[1]
	img_data = img_data.flatten().tolist();

	var resized = new Jimp(height,width, function (err, image) {

		let buffer = image.bitmap.data
		var i = 0;
		for(var x=0; x<height*width*4;x+=4) {
			buffer[x] = img_data[i];
			buffer[x+1] = img_data[i];
			buffer[x+2] = img_data[i];
			buffer[x+3] = 255;
			i++;
		}

		callback(image);

        image.resize(target_height,target_width);	
	});

	var i = 0;
	let resized_image_data = new Float64Array(target_height*target_width);
    for(var x=0; x<height*width*4;x+=4) {
		resized_image_data[i] = resized.bitmap.data[x];
		i++;
		
	}

    return resized_image_data;
}