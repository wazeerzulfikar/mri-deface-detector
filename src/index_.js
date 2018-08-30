var readNifti = require('./readNifti')
var resize = require('./resize')
var preprocess = require('./preprocess')
var predict = require('./predict')

module.exports = {
	readNifti : readNifti,
	resize : resize,
	preprocess : preprocess,
	predict : predict
}
