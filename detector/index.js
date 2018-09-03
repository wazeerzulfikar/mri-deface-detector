// dependencies ------------------------------------------------------

var utils = require('./utils')
var detector = require('./detector')

// public api --------------------------------------------------------

var defaceDetector = {
	readNifti : utils.readNifti,
	loadModel : utils.loadModel,
	detectDeface : detector.detectDeface,
	detectDefaceCustom : detector.detectDefaceCustom
}

// exports -----------------------------------------------------------

module.exports = defaceDetector
