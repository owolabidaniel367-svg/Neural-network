
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
// Neurons
class Neuron {
    constructor(inputCount) {
        this.weights = [];
        this.bias = Math.random();
        this.activation;
        this.activationDerivative;
        this.errorloss;
        this.inputs;
        this.delta;

        for (let i = 0; i < inputCount; i++) {
            this.weights.push(Math.random());
        }
    }

    forward(inputs) {
        let sum = this.bias;
        this.inputs = inputs
        for (let i = 0; i < inputs.length; i++) {
            sum += inputs[i] * this.weights[i];
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
        return this.neurons.map(neuron => neuron.forward(inputs));
    }
       update(learningRate) {

        for (let neuron of this.neurons) {

            neuron.update(learningRate);

        }

    }
    }
  


   


// Neural Network
class NeuralNetwork {
    constructor(inputCount, layerArr, outputCount) {
        this.inputCount = inputCount
        this.layers = [];
        this.expectedOutput;
        this.Prediction
        this.errorloss = []
        this.learningRate = 0.5;

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
run(arr,expectedOutput){
    let cond;
    if (this.inputCount !== arr.length) {
        throw new Error("Input size mismatch");
    }
  this.expectedOutput = expectedOutput
    let output = this.layers[0].forward(arr);
    for (let i = 1; i < this.layers.length; i++) {
        output = this.layers[i].forward(output);
    }
    output = this.outputLayer.forward(output);
    this.Prediction = output
     const predicted = getbest(this.Prediction);
    const actual = getbest(this.expectedOutput);
    const correct = predicted[0] === actual[0];

    return { predicted, correct }
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
train(trainingData, epochs) {

    for (let epoch = 0; epoch < epochs; epoch++) {

        let totalLoss = 0;

        for (let example of trainingData) {

            // Forward propagation
            this.run(example.input,example.output);

            // Calculate loss
            totalLoss += this.Trainingloss(this.Prediction);

            // Backpropagation
            this.Backdrop();

        }

        // Average loss for this epoch
        totalLoss /= trainingData.length;


        if (epoch % 1000 === 0) {

            console.log(
                "Epoch:",
                epoch,
                "Loss:",
                totalLoss);

        }

    }

}
   

}
const trainingData = [
    { input: [0, 0, 0], output: [1, 0] }, // 0 ones -> even
    { input: [0, 0, 1], output: [0, 1] }, // 1 one  -> odd
    { input: [0, 1, 0], output: [0, 1] }, // 1 one  -> odd
    { input: [0, 1, 1], output: [1, 0] }, // 2 ones -> even
    { input: [1, 0, 0], output: [0, 1] }, // 1 one  -> odd
    { input: [1, 0, 1], output: [1, 0] }, // 2 ones -> even
    { input: [1, 1, 0], output: [1, 0] }, // 2 ones -> even
    { input: [1, 1, 1], output: [0, 1] }, // 3 ones -> odd
];

function getbest(arr) {

    let biggest = arr[0];
let biggestIndex = 0;

    for (let i = 1; i < arr.length; i++) {

        if (arr[i] > biggest) {

            biggest = arr[i];
            biggestIndex = i;

        }
    }

    return [biggestIndex, biggest];
}
const neural = new NeuralNetwork(3, [12], 2);

neural.train(trainingData, 300000);
console.log("000:", neural.run([0, 0, 0], [1, 0])); // even
console.log("001:", neural.run([0, 0, 1], [0, 1])); // odd
console.log("010:", neural.run([0, 1, 0], [0, 1])); // odd
console.log("011:", neural.run([0, 1, 1], [1, 0])); // even
console.log("100:", neural.run([1, 0, 0], [0, 1])); // odd
console.log("101:", neural.run([1, 0, 1], [1, 0])); // even
console.log("110:", neural.run([1, 1, 0], [1, 0])); // even
console.log("111:", neural.run([1, 1, 1], [0, 1])); // odd