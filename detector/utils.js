var nj = require('numjs');
var Jimp = require('jimp');
var niftijs = require('nifti-reader-js');
var tf = require('@tensorflow/tfjs');

var utils = {
  /**
   * loadModel
   *
   * Given the filename, the function asynchronously loads
   * the model and the callback handles the response
   */

  loadModel: async function(filename, callback) {
    model = await tf.loadModel(filename);
    callback(model);
    return model;
  },

  /**
   * readNifti
   *
   * The function takes a file, checks for NIFTI, reads it
   * and returns the necessary contents. The callback handles errors.
   */

  readNifti: function(file, callback) {
    if (niftijs.isCompressed(file)) {
      file = niftijs.decompress(file);
      console.log('Decompressed');
    }

    if (niftijs.isNIFTI(file)) {
      var niftiHeader = niftijs.readHeader(file);
      console.log(niftiHeader)
      var dimensions = niftiHeader.dims.slice(1, 4).reverse();
      console.log('Dimensions : ' + dimensions);

      var image = niftijs.readImage(niftiHeader, file);
      var imagePixels = dimensions.reduce((prod, ele)=>prod*ele);

      if (image.byteLength==imagePixels) {
        var imageData = new Int8Array(image);
      } else if (image.byteLength==imagePixels*2){
        var imageData = new Int16Array(image);
      } else if (image.byteLength==imagePixels*4){
        var imageData = new Float32Array(image);
      } else if (image.byteLength==imagePixels*8) {
        var imageData = new Float64Array(image);
      } else {
        callback('Error in file data format!');
        return;
      }

      if(niftiHeader.littleEndian==false) {
        console.log('Need to Fix for Big Endian!');
        callback('Need to Fix for Big Endian!');
        return;
      }

    } else {
      callback(`Error! Please provide a valid NIFTI file.`);
      return;
    }

    return {
      image: imageData,
      dimensions: dimensions
    };
  },

  /**
   * preprocess
   *
   * Takes image and dimensions as read using readNifti, input_size indicating
   * the input shape to the trained model. Preprocess method is one of `slice/mean`.
   * Callback for displaying the image in the browser.
   */

  preprocess: function(
    contents,
    dimensions,
    input_size,
    preprocess_method,
    callback
  ) {
    var img = nj.float64(contents);

    img = utils.minmaxNormalize(img);

    img = nj.uint8(img.reshape(dimensions));

    var slices = [];

    if (preprocess_method == 'slice') {
      for (var i = 0; i < dimensions.length; i++) {
        var slice = utils.centerSlice(img, dimensions, i).T;
        slices.push(slice);
      }
    } else if (preprocess_method == 'mean') {
      for (var i = 0; i < dimensions.length; i++) {
        slice = utils.axisMean(img, dimensions, i).T;
        slices.push(utils.minmaxNormalize(slice));
      }
    }

    for (var i = 0; i < dimensions.length; i++) {
      slices[i] = nj.float64(
        utils.resizeImage(slices[i], input_size, callback)
      );
      slices[i] = nj.divide(slices[i], 255);
    }

    return slices;
  },

  /**
   * resizeImage
   *
   * Takes a grayscale image as a numjs NdArray, and target size.
   * Callback to display the image on the browser.
   * Returns the resized image as FloatArray.
   */

  resizeImage: function(img_data, target_size, callback) {
    console.log('Resizing..');

    var width = img_data.shape[0];
    var height = img_data.shape[1];
    img_data = img_data.flatten().tolist();

    var resized = new Jimp(height, width, function(err, image) {
      let buffer = image.bitmap.data;
      var i = 0;
      for (var x = 0; x < height * width * 4; x += 4) {
        buffer[x] = img_data[i];
        buffer[x + 1] = img_data[i];
        buffer[x + 2] = img_data[i];
        buffer[x + 3] = 255;
        i++;
      }

      callback(image);

      image.resize(target_size, target_size);
    });

    var i = 0;
    let resized_image_data = new Float64Array(target_size * target_size);
    for (var x = 0; x < height * width * 4; x += 4) {
      resized_image_data[i] = resized.bitmap.data[x];
      i++;
    }

    return resized_image_data;
  },

  /**
   * axisMean
   *
   * Preprocess method which takes 3D mri scan as numjs NdArray and the axis,
   * returns the respective slice after performing arithmetic mean preprocess.
   * More information on the readme.
   */

  axisMean: function(img, dimensions, axis) {
    axes = [0, 1, 2];
    axes.splice(axis, 1);

    var slice = [];

    for (var i = 0; i < dimensions[axes[0]]; i++) {
      for (var j = 0; j < dimensions[axes[1]]; j++) {
        key = [null, null, null];
        key[axes[0]] = i;
        key[axes[1]] = j;
        slice.push(img.pick(...key).mean());
      }
    }

    return nj
      .float64(slice)
      .reshape([dimensions[axes[0]], dimensions[axes[1]]]);
  },

  /**
   * centerSlice
   *
   * Preprocess method which takes 3D mri scan as numjs NdArray and the axis,
   * returns respective slices after performing center slice preprocess.
   * More information on the readme.
   */

  centerSlice: function(img, dimensions, axis) {
    var key = [null, null, null];
    key[axis] = dimensions[axis] / 2;
    return img.pick(...key);
  },

  /**
   * minmaxNormalize
   *
   * Takes the image, and does the min max normalization on it. Important for
   * successful nifti read.
   * Returns the normalized image.
   */

  minmaxNormalize: function(img) {
    var max_val = img.max();
    var min_val = img.min();

    img = nj.divide(img, max_val - min_val);
    img = nj.multiply(img, 255);
    if (min_val < 0) {
      img = nj.add(img, 127.5);
    }

    return img;
  },
};

module.exports = utils;
