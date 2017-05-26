const cv = require('opencv');
const Matrix = cv.Matrix;

function toRGBObject(bgr) {
  return {
    r: Math.round(bgr[2]),
    g: Math.round(bgr[1]),
    b: Math.round(bgr[0])
  }
}

function avgVector(arr) {
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

async function imageLoader (filename) {
  return await new Promise((resolve, reject) => {
    cv.readImage(filename, function(err, im) {
      if(err) reject(err);
      resolve(im);
    })
  })
}

function thresholdHue(_im,t) {
  var im = _im.copy();
  im.convertHSVscale();
  im.inRange([t,0,0],[360,255,255])

  return im;
}

async function extractEyeFeature (im) {
  return await new Promise((resolve, reject) => {
    im.detectObject(cv.EYE_CASCADE, {}, function(err, eyes){
      var im_eyes = [];
      var means = [];

      for(var i=0; i<eyes.length; i++) {
        var e = eyes[i];

        var im_eye = im.crop(e.x + e.width/4, e.y + e.height/4, e.width/2, e.height/2);
        var mask = thresholdHue(im_eye,40);

        means[i] = im_eye.meanWithMask(mask);

        im_eye.save('./out_eye_'+i+'.png')
        mask.save('./out_eye_mask_'+i+'.png')
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
  return features;
}

processImage("./img/test.jpg").then((f) => console.log(f));
// cv.readImage("./img/darya.jpg", function(err, im){
//   im.detectObject(cv.FACE_CASCADE, {}, function(err, faces){
//     console.log("Faces: "+faces.length);
//     var im_faces = [];
//     for(var i=0; i<faces.length; i++) {
//       var f = faces[i];
//       im_faces[i] = im.crop(f.x,f.y,f.width,f.height);
//       console.log(im_faces[i].mean());
//
//       // im.rectangle([f.x, f.y], [f.width, f.height], [0,0,255] ,1);
//     }
    // im.detectObject(cv.EYE_CASCADE, {}, function(err, eyes){
    //   console.log("Eyes: "+eyes.length);
    //   var im_eyes = [];
    //
    //   for(var i=0; i<eyes.length; i++) {
    //     var e = eyes[i];
    //
    //     im_eyes[i] = im.crop(e.x + e.width/4,
    //                          e.y + e.height/4,
    //                          e.width/2,
    //                          e.height/2);
    //
    //     // im_eyes[i].convertHSVscale();
    //     // console.log(im_eyes[i].mean());
    //     // im_eyes[i].
    //     // im_eyes[i].dilate(5);
    //     // im.rectangle([e.x, e.y], [e.width, e.height], [255,0,0] ,1);
    //   }
//
//       for(var i=0; i<im_faces.length; i++) {
//         im_faces[i].save('.out_face_'+i+'.jpg')
//       }
//
//       for(var i=0; i<im_eyes.length; i++) {
        // im_eyes[i].save('.out_eye_'+i+'.jpg')
//       }
//
//       im.convertHSVscale();
//       im.save('./out.jpg');
//     })
//   })
// })
