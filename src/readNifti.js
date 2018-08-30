var niftijs = require('nifti-reader-js')

/**
 * Read a nifti file given the file
 *
 * @param {string} filename
 * @param {function} 
 * @return {FloatArray}
 */

module.exports = function (file, callback) {

		if (niftijs.isCompressed(file)) {
			var file = niftijs.decompress(file);
			console.log('Decompressed')
		}

		if (niftijs.isNIFTI(file)) {
			var niftiHeader = niftijs.readHeader(file);
			var dimensions = niftiHeader.dims.slice(1,4).reverse();
			console.log('Dimensions : '+dimensions);
			var image = niftijs.readImage(niftiHeader, file);
		} else {
			callback(`Error! Please provide a valid NIFTI file.`);
			return;
		}

		return {
			image: new Int16Array(image),
			dimensions : dimensions
		};

	}

