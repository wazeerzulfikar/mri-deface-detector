// dependencies ------------------------------------------------------

var utils = require('./utils')
var predict = require('./predict')

// public api --------------------------------------------------------

var defaceDetector = {
	readNifti : utils.readNifti,
	loadModel : utils.loadModel,
	predict : predict
}

// exports -----------------------------------------------------------

module.exports = defaceDetector
