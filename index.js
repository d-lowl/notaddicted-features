const cv = require('opencv');
const Matrix = cv.Matrix;

async function imageLoader (filename) {
  return await new Promise((resolve, reject) => {
    cv.readImage(filename, function(err, im) {
      if(err) reject(err);
      resolve(im);
    })
  })
}

function toRGBObject(bgr) {
  if(!bgr) {
    return null;
  }
  return {
    r: Math.round(bgr[2]),
    g: Math.round(bgr[1]),
    b: Math.round(bgr[0])
  }
}

function avgVector(arr) {
  if(arr.length == 0)
  {
    return null;
  }

  var vector = new Array(arr[0].length).fill(0);

  var mul = 1.0/arr.length;
  for(var i = 0; i < arr.length; i++)
  {
    for(var j = 0; j < arr[i].length; j++)
    {
      vector[j] += arr[i][j] * mul;
    }
  }
  return vector;
}

function thresholdHue(_im,t1,t2) {
  var im = _im.copy();
  im.convertHSVscale();
  im.inRange([t1,0,0],[t2,255,255])

  return im;
}

async function extractEyeFeature (im) {
  return await new Promise((resolve, reject) => {
    im.detectObject(cv.EYE_CASCADE, {}, function(err, eyes){
      var means = [];

      for(var i=0; i<eyes.length; i++) {
        var e = eyes[i];
        var im_eye = im.crop(e.x + e.width/4, e.y + e.height/4, e.width/2, e.height/2);
        var mask = thresholdHue(im_eye,45,360);

        means[i] = im_eye.meanWithMask(mask);

        im_eye.save('./out_eye_'+i+'.png')
        mask.save('./out_eye_mask_'+i+'.png')
      }
      var avgMean = avgVector(means);
      resolve(toRGBObject(avgMean));
    })
  })
}

async function extractSkinFeature (im) {
  return await new Promise((resolve, reject) => {
    im.detectObject(cv.FACE_CASCADE, {}, function(err, faces){
      var means = [];

      for(var i=0; i<faces.length; i++) {
        var f = faces[i];

        var im_face = im.crop(f.x + f.width/4, f.y + f.height/4, f.width/2, f.height/2);
        var mask = thresholdHue(im_face,0,50);

        means[i] = im_face.meanWithMask(mask);

        im_face.save('./out_face_'+i+'.png')
        mask.save('./out_face_mask_'+i+'.png')
      }
      var avgMean = avgVector(means);
      resolve(toRGBObject(avgMean));
    })
  })
}

async function processImage (filename) {
  var features = {};
  var im = await imageLoader(filename);
  features.eyes = await extractEyeFeature(im);
  features.skin = await extractSkinFeature(im);
  return features;
}

processImage(process.argv[2]).then((f) => console.log(f));
