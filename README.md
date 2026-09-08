# Neural Network from Scratch (JavaScript)

A feedforward neural network built from scratch in plain JavaScript — no ML libraries — trained on the MNIST handwritten digit dataset. Includes a custom binary parser for the IDX file format.

## Project Structure

```
Neural-network/
├── data/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── parser.js          # Parses MNIST IDX files into training data
└── Neural Network.js  # Neuron/Layer/NeuralNetwork classes + training loop
```

## How It Works

### 1. Parsing (`parser.js`)

MNIST files use the IDX binary format (big-endian):

- **Labels file**: 8-byte header (magic number, item count) followed by one byte per label (0–9).
- **Images file**: 16-byte header (magic number, image count, rows, cols) followed by `rows × cols` bytes per image, flattened row-by-row.

`parser.js` decompresses the `.gz` files with `zlib`, reads the headers with `buffer.readUInt32BE()`, extracts each label/pixel with `buffer.readUInt8()`, normalizes pixel values to `0–1`, and one-hot encodes labels (digit `3` → `[0,0,0,1,0,0,0,0,0,0]`).

Exports a single function:

```js
export function getTrainingData() {
  // returns [{ input: [...784 pixels], output: [...10 one-hot] }, ...]
}
```

### 2. Network (`Neural Network.js`)

Three classes:

- **`Neuron`** — holds `weights` + `bias`, computes a sigmoid-activated output (`forward`), and updates itself via gradient descent (`update`).
- **`Layer`** — a collection of neurons; forwards inputs through all of them at once.
- **`NeuralNetwork`** — chains layers together (hidden layers + output layer), and handles training via backpropagation.

```js
const neural = new NeuralNetwork(inputCount, hiddenLayerSizes, outputCount);
```

Example:

```js
const neural = new NeuralNetwork(784, [16, 16], 10);
```

- `784` — input size (28×28 pixels)
- `[16, 16]` — two hidden layers, 16 neurons each (array length = number of hidden layers, each value = neuron count for that layer)
- `10` — output size (digits 0–9)

### 3. Training

```js
neural.train(trainingData, epochs);
```

Runs forward propagation, computes loss (mean squared error), backpropagates the error, and updates every weight/bias — once per training example, per epoch (stochastic gradient descent).

**Note on speed:** training time scales with `examples × epochs × network size`. With the full 60,000-example MNIST set, use a modest epoch count (10–30), not the hundreds of thousands used for tiny toy datasets. Test on a small slice first:

```js
const smallData = trainingData.slice(0, 1000);
neural.train(smallData, 20);
```

### 4. Running / Predicting

```js
const result = neural.run(example.input, example.output);
// { predicted: [digit, confidence], correct: true/false }
```

## Usage

```js
import { getTrainingData } from "./parser.js";

const trainingData = getTrainingData();
const neural = new NeuralNetwork(784, [16, 16], 10);

neural.train(trainingData, 20);

console.log(neural.run(trainingData[0].input, trainingData[0].output));
```

Run with:

```bash
node "Neural Network.js"
```

## Requirements

- Node.js
- `"type": "module"` set in `package.json` (this project uses ES module `import`/`export` syntax)
- MNIST `.gz` files placed in `./data/`

## Saving / Loading Trained Weights

To avoid retraining every run, export learned weights/biases to JSON and reload them into a freshly constructed network of the same architecture. See `toJSON()` / `loadWeights()` methods on `NeuralNetwork` (add these if not already present) for the save/load pattern.

## Status / Known Limitations

- Pure JavaScript, no vectorization — training on the full dataset is slow. For serious speed, a matrix-based (vectorized) rewrite or a compiled language would help far more than tuning hyperparameters alone.
- Architecture (input/hidden/output sizes) must match exactly between save and load — `loadWeights` only restores values, not structure.
- Future changes, Would use tensor flow to optimize matrix multiplications and improve compute.
