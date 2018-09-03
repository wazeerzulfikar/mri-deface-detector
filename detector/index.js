// dependencies ------------------------------------------------------

var utils = require('./utils')
var detector = require('./detector')

// public api --------------------------------------------------------

var defaceDetector = {
	readNifti : utils.readNifti,
	loadModel : utils.loadModel,
	detect : detector.defaceDetect,
	customPredict : detector.customPredict
}

// exports -----------------------------------------------------------

module.exports = defaceDetector
