import { getTrainingData,getTestingData } from "./parser.js";
import fs from 'fs';
const trainingData = getTrainingData()
const TestingData = getTestingData()
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
// Neurons

class Neuron {
    constructor(inputCount) {
        this.weights = new Float32Array(inputCount);
        this.bias = 0
        this.activation;
        this.activationDerivative;
        this.errorloss;
        this.inputs;
        this.delta;
      

   const limit = Math.sqrt(6 / inputCount)

for (let i = 0; i < inputCount; i++) {
    this.weights[i] = (
        (Math.random() * 2 - 1) * limit
    )
}
    }
    forward(inputs) {
        let sum = this.bias;
        this.inputs = inputs
       const weights = this.weights;
const length = weights.length;

for (let i = 0; i < length; i++) {
    sum += inputs[i] * weights[i];
}
        this.activation = sigmoid(sum)
        this.activationDerivative = this.activation * (1-this.activation)
        
        return this.activation;
    }
     update(learningRate) {

        this.bias -=
            learningRate * this.delta;


        for (let i = 0; i < this.weights.length; i++) {

            const gradient = this.delta * this.inputs[i];
            this.weights[i] -= learningRate * gradient;

        }

    }

}

// Layer
class Layer {
    constructor(inputCount,neuronCount) {
        this.neurons = []
        this.errorlosses = []
        this.inputCount = inputCount
        this.weightmat = []

    
    for (let i = 0; i < neuronCount; i++) {
        this.neurons.push(new Neuron(inputCount))
        
    }
 
    }
     forward(inputs) {
          this.inputs = inputs;

    const outputs =  new Float32Array(this.neurons.length);

    for (let i = 0; i < this.neurons.length; i++) {
        outputs[i] = this.neurons[i].forward(inputs);
    }

    return outputs;
}
    
       update(learningRate) {

        for (let neuron of this.neurons) {

            neuron.update(learningRate);

        }

    }
    }
  


   


// Neural Network
export default class NeuralNetwork {
    constructor(inputCount, layerArr, outputCount) {
        this.inputCount = inputCount
        this.layers = [];
        this.expectedOutput;
        this.Prediction
        this.errorloss = []
        this.learningRate = 0.001;
        
        for (let i = 0; i < layerArr.length; i++) {

            if (i === 0) {
                this.layers.push(
                    new Layer(inputCount, layerArr[i])
                );
            } else {
                this.layers.push(
                    new Layer(layerArr[i - 1], layerArr[i])
                );
            }
        }

        this.outputLayer = new Layer(
            layerArr[layerArr.length - 1],
            outputCount
        );
    }
run(arr, expectedOutput = this.expectedOutput) {
    if (this.inputCount !== arr.length) {
        throw new Error("Input size mismatch");
    }
    this.expectedOutput = expectedOutput;
    let output = this.layers[0].forward(arr);
    for (let i = 1; i < this.layers.length; i++) {
        output = this.layers[i].forward(output);
    }
    output = this.outputLayer.forward(output);
    this.Prediction = output;

    return output;
}

saveModel(filename) {

    const model = {
        inputCount: this.inputCount,

        layers: this.layers.map(layer =>
            layer.neurons.map(neuron => ({
                weights: Array.from(neuron.weights),
                bias: neuron.bias
            }))
        ),

        outputLayer: this.outputLayer.neurons.map(neuron => ({
            weights: Array.from(neuron.weights),
            bias: neuron.bias
        }))
    };

    fs.writeFileSync(
        filename,
        JSON.stringify(model)
    );

    console.log("Model saved successfully!");
}
loadModel(filename) {

    const model = JSON.parse(
        fs.readFileSync(filename, "utf8")
    );

    for (let i = 0; i < this.layers.length; i++) {

        for (let j = 0; j < this.layers[i].neurons.length; j++) {

            const neuron = this.layers[i].neurons[j];
            const saved = model.layers[i][j];

            neuron.weights.set(saved.weights);
            neuron.bias = saved.bias;
        }
    }

    for (let i = 0; i < this.outputLayer.neurons.length; i++) {

        const neuron = this.outputLayer.neurons[i];
        const saved = model.outputLayer[i];

        neuron.weights.set(saved.weights);
        neuron.bias = saved.bias;
    }

    console.log("Model loaded successfully!");
}
Backdrop(){

   // Neural net for each output neuron 
  
   // backdrop for the output layer
  for (let i = 0; i < this.outputLayer.neurons.length; i++) {

    const neuron = this.outputLayer.neurons[i];

    const error = neuron.activation - this.expectedOutput[i];

    neuron.delta = error * neuron.activationDerivative;
}
   
   // Backdrop for all the layers
  for (let layerIndex = this.layers.length - 1; layerIndex >= 0; layerIndex--) {

    const currentLayer = this.layers[layerIndex];

    const nextLayer =
        layerIndex === this.layers.length - 1
            ? this.outputLayer
            : this.layers[layerIndex + 1];


    for (let i = 0; i < currentLayer.neurons.length; i++) {

        const neuron = currentLayer.neurons[i];

        let propagatedError = 0;


        for (let j = 0; j < nextLayer.neurons.length; j++) {

            const nextNeuron = nextLayer.neurons[j];

            propagatedError +=
                nextNeuron.weights[i] *
                nextNeuron.delta;
        }


        neuron.delta =
            propagatedError *
          
          neuron.activationDerivative;
    }
}
// Update the weight and bias for all the layers both out put and hidden layers
this.outputLayer.update(this.learningRate)
for (let lay = 0; lay < this.layers.length; lay++) {
        this.layers[lay].update(this.learningRate)
    
}

}
Trainingloss(output){
    let sum = 0
    for (let i = 0; i < output.length; i++) {
        const  error =   this.expectedOutput[i] - output[i]
        sum +=(error**2)/2
        
    }
    sum = sum/output.length
    return sum
}
test(testingData){
    testingData = shuffle(testingData)
    let epochs = testingData.length
    let wrongCount = 0
    let rightCount = 0
    let value 
    let numbersfailed = [0,0,0,0,0,0,0,0,0,0]
    for (let epoch = 0; epoch < epochs; epoch++) {
        let loss = 0
       
        let TestOutput = testingData[epoch].output
        let Num = testingData[epoch].outputNum
         this.run(testingData[epoch].input, TestOutput);   
           loss =   this.Trainingloss(this.Prediction);
           let predIndex = getbest(this.Prediction)
           let outIndex = getbest(TestOutput)
           if(predIndex === outIndex){
            value = "True"
            rightCount = rightCount + 1
           }
           else{
            value = "Wrong"
            wrongCount = wrongCount + 1
            numbersfailed[Num] = numbersfailed[Num] + 1
           }
                 if (epoch % 10 === 0) {

            console.log(
                "Epoch:",
                epoch,
                "Loss:",
                loss,
                value  );

        }
           }
           let accuracy = (rightCount/epochs) * 100
           console.log(`This model got ${rightCount} right and ${wrongCount} wrong. Accuracy: ${accuracy}%`)
           console.log(numbersfailed)
}
train(trainingData, epochs) {
       trainingData = shuffle(trainingData)
       let index = 0
    for (let epoch = 0; epoch < epochs; epoch++) {

        let Loss = 0;
        

      
            // Forward propagation
            this.run(trainingData[index].input,trainingData[index].output);

            // Calculate loss
            Loss= this.Trainingloss(this.Prediction);

            // Backpropagation
            this.Backdrop();



        if (epoch % 10 === 0) {
           let percentageDone = ((epoch/epochs) * 100)
           let fixedPerecntage = percentageDone.toFixed(3)
            console.log(
                "Epoch:",
                epoch,
                "Loss:",
                Loss,
                 "Done:",
                fixedPerecntage);

        }
          index = (index+1)% trainingData.length
    }

}
   

}


function getbest(arr) {

    let biggest = arr[0];
let biggestIndex = 0;

    for (let i = 1; i < arr.length; i++) {

        if (arr[i] > biggest) {

            biggest = arr[i];
            biggestIndex = i;

        }
    }

    return biggestIndex
}
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));

        [array[i], array[j]] = [array[j], array[i]];
    }

    return array;
}
const neural = new NeuralNetwork(784, [16,16],10 );

function train(){
 neural.loadModel("MNIST2.json")

neural.train(trainingData, 1000000);
 neural.saveModel("MNIST2.json")
}
function test(){
     neural.loadModel("MNIST2.json")
neural.test(TestingData)
}
function Modelmode(mode){
    if(mode === "train"){
        train()
    }
    if(mode === "test"){
        test()
    }
}

Modelmode("train")
Modelmode("test")

