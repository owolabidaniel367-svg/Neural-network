// Training labels 
import fs from 'fs';
import zlib from 'zlib';

const compressed = fs.readFileSync('./data/train-labels-idx1-ubyte.gz');
const buffer = zlib.gunzipSync(compressed);

const magic = buffer.readUInt32BE(0);
const numLabels = buffer.readUInt32BE(4);
console.log(magic, numLabels);

const labels = [];
for (let i = 0; i < numLabels; i++) {
  labels.push(buffer.readUInt8(8 + i));
}

// Trainig images
const imgCompresed = fs.readFileSync('./data/train-images-idx3-ubyte.gz')
const imagesBuffer = zlib.gunzipSync(imgCompresed);
const count = imagesBuffer.readUInt32BE(4)
const rows = imagesBuffer.readUInt32BE(8)
const cols = imagesBuffer.readUInt32BE(12)

const images = [];
for (let i = 0; i < count; i++) {
  let image = [];
  for (let t = 0; t < 784; t++) {
    const pixelValue = imagesBuffer.readUInt8(16 + i * rows * cols + t);
    image.push(pixelValue/255);
  }
  images.push(image);
}

const trainingData = []
export function getTrainingData(){
for (let i = 0; i < count; i++) {
    const input = images[i]
    const output = new Array(10).fill(0)
    output[labels[i]] = 1
    trainingData.push({input,output})
   
}
 return trainingData
}
// Testing data code
// Testing labels
const testLabelsCompressed = fs.readFileSync('./data/t10k-labels-idx1-ubyte.gz');
const testLabelsBuffer = zlib.gunzipSync(testLabelsCompressed);
const testNumLabels = testLabelsBuffer.readUInt32BE(4);

const testLabels = [];
for (let i = 0; i < testNumLabels; i++) {
  testLabels.push(testLabelsBuffer.readUInt8(8 + i));
}

// Testing images
// Testing images
const testImgCompressed = fs.readFileSync('./data/t10k-images-idx3-ubyte.gz')
const testImagesBuffer = zlib.gunzipSync(testImgCompressed);
const testCount = testImagesBuffer.readUInt32BE(4)
const testRows = testImagesBuffer.readUInt32BE(8)
const testCols = testImagesBuffer.readUInt32BE(12)

const testImages = [];
for (let i = 0; i < testCount; i++) {
  let image = [];
  for (let t = 0; t < 784; t++) {
    const pixelValue = testImagesBuffer.readUInt8(16 + i * testRows * testCols + t);
    image.push(pixelValue/255);
  }
  testImages.push(image);
}
export function getTestingData() {
  const testingData = [];
  for (let i = 0; i < testCount; i++) {
    const output = new Array(10).fill(0);
    output[testLabels[i]] = 1;
    testingData.push({ input: testImages[i], output });
  }
  return testingData;
}
console.log(testLabels.length)