# Neural Network from Scratch in JavaScript

A fully connected feedforward neural network implemented from first principles in plain JavaScript, without machine-learning frameworks or numerical-computing libraries.

The project implements the core mathematical and computational components of a neural network, including:

- Neurons
- Dense layers
- Weighted sums
- Biases
- Sigmoid activation
- Forward propagation
- Mean Squared Error (MSE)
- Backpropagation
- Chain-rule gradient computation
- Gradient descent
- Stochastic Gradient Descent (SGD)
- MNIST IDX binary parsing
- Gzip decompression
- Pixel normalization
- One-hot encoding
- Digit classification
- Parameter serialization

The primary objective is not maximum performance. It is to understand how neural networks work internally by implementing the underlying mathematics directly.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [MNIST Dataset](#mnist-dataset)
- [IDX Binary Format](#idx-binary-format)
- [Label File Format](#label-file-format)
- [Image File Format](#image-file-format)
- [Input Representation](#input-representation)
- [One-Hot Encoding](#one-hot-encoding)
- [Neuron](#neuron)
- [Activation Function](#activation-function)
- [Dense Layer](#dense-layer)
- [Forward Propagation](#forward-propagation)
- [Computational Graph](#computational-graph)
- [Loss Function](#loss-function)
- [Backpropagation](#backpropagation)
- [Chain Rule](#chain-rule)
- [Backpropagation Through Multiple Layers](#backpropagation-through-multiple-layers)
- [Gradient Descent](#gradient-descent)
- [Why the Negative Gradient?](#why-the-negative-gradient)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [Training Process](#training-process)
- [Training Example](#training-example)
- [Prediction](#prediction)
- [Output Activation and Confidence](#output-activation-and-confidence)
- [Parameter Count](#parameter-count)
- [Computational Complexity](#computational-complexity)
- [Memory Complexity](#memory-complexity)
- [Serialization](#serialization)
- [Running the Project](#running-the-project)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Design Philosophy](#design-philosophy)
- [Implementation Status](#implementation-status)
- [Educational Scope](#educational-scope)
- [Mathematical Summary](#mathematical-summary)
- [Conclusion](#conclusion)

---

## Architecture

The neural network is a composition of fully connected layers, for example:

```text
784 → 16 → 16 → 10
```

```js
const neural = new NeuralNetwork(784, [16, 16], 10);
```

```text
Input Layer (784 neurons)
        │
        ▼
Hidden Layer (16 neurons)
        │
        ▼
Hidden Layer (16 neurons)
        │
        ▼
Output Layer (10 neurons)
```

The 784 inputs correspond to the pixels of a 28 × 28 MNIST image. The 10 output neurons correspond to the digits `0` through `9`.

---

## Project Structure

```text
Neural-network/
│
├── data/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
│
├── parser.js
├── Neural Network.js
├── package.json
└── README.md
```

### `parser.js`

Converts the raw MNIST binary files into JavaScript data structures:

1. Gzip decompression
2. IDX header parsing
3. Image extraction
4. Label extraction
5. Pixel normalization
6. One-hot encoding
7. Training-example construction

The resulting structure is approximately:

```js
{
    input: [...784 pixel values],
    output: [...10 one-hot encoded values]
}
```

### `Neural Network.js`

Contains the neural-network implementation, divided conceptually into three levels:

```text
Neuron → Layer → NeuralNetwork
```

- **Neuron** — an individual computational unit
- **Layer** — a collection of neurons
- **NeuralNetwork** — connects layers and performs forward propagation, loss calculation, backpropagation, parameter updates, and prediction

---

## MNIST Dataset

The network is trained on the MNIST handwritten digit dataset. Each example consists of a 28 × 28 grayscale image and a digit label, giving $28 \times 28 = 784$ pixel values per image.

The dataset contains 60,000 training images and 10,000 test images.

---

## IDX Binary Format

MNIST is distributed using the IDX binary file format:

```text
.gz → gzip decompression → IDX binary data → header parsing → images / labels
```

The parser uses Node.js `zlib` to decompress the files.

---

## Label File Format

The label file contains an 8-byte header:

| Field | Size |
|---|---|
| Magic Number | 4 bytes |
| Item Count | 4 bytes |

Values are stored in big-endian byte order and read with `buffer.readUInt32BE()`. After the header, each label occupies one byte, read with `buffer.readUInt8()`.

---

## Image File Format

The image file contains a 16-byte header:

| Field | Size |
|---|---|
| Magic Number | 4 bytes |
| Image Count | 4 bytes |
| Rows | 4 bytes |
| Columns | 4 bytes |

For MNIST, rows $= 28$ and columns $= 28$, so each image contains $28 \times 28 = 784$ pixels, each occupying one byte.

---

## Input Representation

Raw pixel values are integers $0 \le p \le 255$. The network normalizes them to $[0, 1]$ using:

$$x = \frac{p}{255}$$

For example: `0 → 0`, `128 → 0.5019...`, `255 → 1`.

The 28 × 28 image is flattened into a 784-value vector:

$$\mathbf{x} \in [0, 1]^{784}$$

---

## One-Hot Encoding

The original MNIST label is an integer 0–9. The network represents this as a 10-dimensional vector with exactly one element equal to 1.

Digit `3`:

$$[0,0,0,1,0,0,0,0,0,0]$$

Digit `7`:

$$[0,0,0,0,0,0,0,1,0,0]$$

In general, $\mathbf{y} \in \{0, 1\}^{10}$.

---

## Neuron

The fundamental computational unit of the network. A neuron holds weights, a bias, an activation function, an activation derivative, and gradient information.

Given input $\mathbf{x} = [x_1, x_2, \dots, x_n]$ and weights $\mathbf{w} = [w_1, w_2, \dots, w_n]$, the neuron computes its pre-activation:

$$z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$

The result is then passed through an activation function.

---

## Activation Function

This implementation uses the sigmoid activation function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad a = \sigma(z)$$

Sigmoid maps arbitrary real values into $(0, 1)$. Its derivative is:

$$\sigma'(z) = \sigma(z)\left(1 - \sigma(z)\right) = a(1 - a)$$

This identity is useful during backpropagation because the neuron already has its activation value cached.

---

## Dense Layer

A dense layer contains multiple neurons; every neuron receives every output from the previous layer.

```text
                 ┌── Neuron 1
Input ───────────┼── Neuron 2
                 ├── Neuron 3
                 └── Neuron m
```

If the previous layer has $n$ values and the current layer has $m$ neurons, the weight matrix has dimensions $m \times n$:

$$\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$

where $W^{(l)}$ is the weight matrix, $\mathbf{a}^{(l-1)}$ is the previous activation vector, $\mathbf{b}^{(l)}$ is the bias vector, and $\mathbf{z}^{(l)}$ is the pre-activation vector. The activation is then:

$$\mathbf{a}^{(l)} = \sigma\!\left(\mathbf{z}^{(l)}\right)$$

---

## Forward Propagation

Forward propagation evaluates the network from the input layer toward the output layer. The first layer receives $\mathbf{a}^{(0)} = \mathbf{x}$. For every subsequent layer:

$$\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \sigma\!\left(\mathbf{z}^{(l)}\right)$$

The final activation vector is $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$, where $L$ is the output layer.

```text
Input → Weighted Sum → Sigmoid → Hidden Layer → Weighted Sum → Sigmoid → Output
```

---

## Computational Graph

A single neuron can be represented as:

```text
x₁ ── × w₁ ──┐
x₂ ── × w₂ ──┤
x₃ ── × w₃ ──┼── Σ ── + b ── sigmoid ──→ a
xₙ ── × wₙ ──┘
```

$$x_i \rightarrow w_i x_i \rightarrow \sum_i w_i x_i + b \rightarrow \sigma(z) \rightarrow a$$

Forward propagation evaluates this graph left to right. Backpropagation propagates gradient information right to left.

---

## Loss Function

The current implementation uses Mean Squared Error. For target vector $\mathbf{y}$ and predicted vector $\hat{\mathbf{y}}$:

$$L = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

For MNIST, $n = 10$:

$$L = \frac{1}{10}\sum_{i=1}^{10} (y_i - \hat{y}_i)^2$$

Training minimizes $L$ with respect to the network parameters.

---

## Backpropagation

Backpropagation computes the derivatives of the loss with respect to the network parameters. For a weight $w_i$, by the chain rule:

$$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a}\cdot\frac{\partial a}{\partial z}\cdot\frac{\partial z}{\partial w_i}$$

Since $a = \sigma(z)$, $\dfrac{\partial a}{\partial z} = a(1-a)$. Since $z = \sum_i w_i x_i + b$, $\dfrac{\partial z}{\partial w_i} = x_i$. Therefore:

$$\boxed{\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a}\, a(1-a)\, x_i}$$

For the bias, $\dfrac{\partial z}{\partial b} = 1$, so:

$$\boxed{\frac{\partial L}{\partial b} = \frac{\partial L}{\partial a}\, a(1-a)}$$

---

## Chain Rule

The chain rule is the mathematical foundation of backpropagation. Suppose $L = L(a)$, $a = a(z)$, and $z = z(w)$. Then:

$$\frac{dL}{dw} = \frac{dL}{da}\cdot\frac{da}{dz}\cdot\frac{dz}{dw}$$

A neural network contains many nested functions of this form:

$$x \rightarrow z^{(1)} \rightarrow a^{(1)} \rightarrow z^{(2)} \rightarrow a^{(2)} \rightarrow L$$

Backpropagation efficiently applies the chain rule through this computational graph.

---

## Backpropagation Through Multiple Layers

Define the error signal $\delta^{(l)} = \dfrac{\partial L}{\partial \mathbf{z}^{(l)}}$.

For the output layer:

$$\delta^{(L)} = \frac{\partial L}{\partial \mathbf{a}^{(L)}} \odot \sigma'\!\left(\mathbf{z}^{(L)}\right)$$

where $\odot$ denotes element-wise multiplication. For a hidden layer:

$$\delta^{(l)} = \left(W^{(l+1)T} \delta^{(l+1)}\right) \odot \sigma'\!\left(\mathbf{z}^{(l)}\right)$$

The weight and bias gradients are:

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \left(\mathbf{a}^{(l-1)}\right)^T, \qquad \frac{\partial L}{\partial \mathbf{b}^{(l)}} = \delta^{(l)}$$

This allows gradient information to propagate backward through an arbitrary number of layers.

---

## Gradient Descent

After calculating the gradients, the parameters are updated:

$$w \leftarrow w - \eta \frac{\partial L}{\partial w}, \qquad b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

where $\eta$ is the learning rate. In vector form:

$$\boxed{\theta_{t+1} = \theta_t - \eta \nabla_\theta L}$$

where $\theta$ represents all trainable parameters and $\nabla_\theta L$ is the gradient.

---

## Why the Negative Gradient?

The gradient $\nabla L$ points in the direction of greatest local increase of the loss, so $-\nabla L$ points toward the greatest local decrease. Gradient descent performs:

$$\theta \leftarrow \theta - \eta \nabla L$$

to move the parameters toward lower-loss regions of the parameter space.

---

## Stochastic Gradient Descent

The current implementation updates the network after processing individual training examples:

```text
for each epoch:
    for each example:
        forward propagation
        calculate loss
        backpropagate
        update weights
        update biases
```

Because parameters are updated after each example, this is a form of Stochastic Gradient Descent. For training example $t$:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L_t$$

where $L_t$ is the loss for the current example.

---

## Training Process

```text
MNIST .gz files
      │
      ▼
Gzip decompression
      │
      ▼
IDX parsing
      │
      ▼
Pixel extraction → Normalization → One-hot encoding
      │
      ▼
Training example
      │
      ▼
Forward propagation → Prediction → Loss calculation
      │
      ▼
Backpropagation → Gradient calculation → Gradient descent
      │
      ▼
Updated parameters ──► Next example
```

One epoch is one complete pass through the training dataset. If $N$ is the number of examples and $E$ the number of epochs, the number of example-level updates is approximately $N \times E$.

For the 60,000-example MNIST training set with 20 epochs:

$$60{,}000 \times 20 = 1{,}200{,}000$$

individual training updates.

---

## Training Example

```js
const neural = new NeuralNetwork(784, [16, 16], 10);

neural.train(trainingData, 20);
```

A smaller dataset can be used for initial testing:

```js
const smallData = trainingData.slice(0, 1000);
neural.train(smallData, 20);
```

This is useful for verifying the implementation before running a full training process.

---

## Prediction

After training, the network produces ten output activations. The predicted class is:

$$\hat{c} = \operatorname*{arg\,max}_{i} \hat{y}_i$$

For example, the output:

```text
[0.01, 0.02, 0.03, 0.91, 0.01, 0.00, 0.01, 0.00, 0.01, 0.01]
```

has its maximum at index `3`, so `prediction = 3`.

```js
const result = neural.run(example.input, example.output);
```

which can return:

```js
{
    predicted: [3, 0.91],
    correct: true
}
```

---

## Output Activation and Confidence

The highest sigmoid activation can be interpreted as an informal confidence score — e.g. `prediction = 3`, `confidence ≈ 0.91`. However, this is **not a calibrated probability**, since sigmoid outputs are independent and $\sum_i \hat{y}_i$ is not constrained to equal 1.

For a proper multi-class probability distribution, a softmax output layer would be more appropriate:

$$P(y_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

which guarantees $\sum_i P(y_i) = 1$.

---

## Parameter Count

For a fully connected layer with $n$ input neurons and $m$ output neurons:

$$\text{weights} = n \times m, \qquad \text{biases} = m, \qquad P = nm + m$$

For the architecture `784 → 16 → 16 → 10`:

| Layer | Parameters |
|---|---|
| Input → Hidden 1 | $784 \times 16 + 16 = 12{,}560$ |
| Hidden 1 → Hidden 2 | $16 \times 16 + 16 = 272$ |
| Hidden 2 → Output | $16 \times 10 + 10 = 170$ |

$$12{,}560 + 272 + 170 = \boxed{13{,}002}$$

trainable parameters.

---

## Computational Complexity

For a dense layer with $n$ inputs and $m$ neurons, the forward pass requires approximately $O(nm)$ operations. For a network with $L$ layers:

$$O\!\left(\sum_{l=1}^{L} n_{l-1} n_l\right)$$

For training over $N$ examples and $E$ epochs, the approximate complexity is:

$$O\!\left(NE \sum_{l=1}^{L} n_{l-1} n_l\right)$$

The exact runtime depends on implementation details, JavaScript engine behavior, memory access, and the cost of the activation and gradient calculations.

---

## Memory Complexity

Parameter storage is approximately:

$$O(P), \qquad P = \sum_{l=1}^{L} \left(n_{l-1} n_l + n_l\right)$$

For a single training example, intermediate activations require approximately:

$$O\!\left(\sum_{l=0}^{L} n_l\right)$$

additional storage. Backpropagation requires these intermediate values because the gradients depend on activations generated during the forward pass.

---

## Serialization

The learned parameters can be serialized so a trained model can be saved and restored. A serialized model contains the learned weights and biases; the architecture must remain compatible — parameters for `784 → 16 → 16 → 10` cannot be loaded into `784 → 32 → 10`, since the dimensions differ.

A serialization interface can use methods such as `toJSON()` and `loadWeights()` to save and restore learned parameters.

---

## Running the Project

### Requirements

- Node.js
- ES module support
- MNIST dataset
- `"type": "module"` in `package.json`

```json
{
    "type": "module"
}
```

### Dataset

Place the MNIST files inside `./data/`:

```text
data/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

### Running

```bash
node "Neural Network.js"
```

---

## Limitations

1. **No vectorization** — operations use ordinary arrays and loops rather than matrix/tensor operations ($Z = WX + b$).
2. **No GPU acceleration** — runs on ordinary JS execution; scaling to larger architectures would be inefficient.
3. **MSE instead of cross-entropy** — categorical cross-entropy with softmax is generally more appropriate for multi-class classification: $L = -\sum_i y_i \log(\hat{y}_i)$.
4. **Sigmoid throughout the network** — deep networks commonly use ReLU-family activations in hidden layers: $\operatorname{ReLU}(x) = \max(0, x)$.
5. **No mini-batch training** — parameters update after individual examples rather than aggregated batches.
6. **Basic SGD optimizer** — no momentum, RMSProp, or Adam.
7. **No regularization** — no L1/L2, dropout, or early stopping.
8. **Limited evaluation pipeline** — no formal train/validation/test split.

---

## Future Improvements

- **Vectorized matrix operations** — replace scalar loops with $Z = WX + b$, $A = \sigma(Z)$.
- **Mini-batch gradient descent** — average gradients across batches before updating.
- **Softmax output layer** — $P(y_i) = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$.
- **Cross-entropy loss** — $L = -\sum_i y_i \log(\hat{y}_i)$.
- **ReLU** — $f(x) = \max(0, x)$ for hidden layers.
- **Xavier / He initialization** — better weight initialization schemes.
- **Adam optimizer** — moving estimates $m_t$, $v_t$ for adaptive updates.
- **Evaluation metrics** — training/validation/test accuracy and loss, confusion matrix, per-class accuracy.
- **Model checkpoints** — save automatically whenever validation performance improves.
- **GPU acceleration** — offload large matrix operations to a tensor library or GPU backend.

---

## Design Philosophy

The project intentionally avoids high-level machine-learning abstractions. Instead of calling `model.fit(...)`, the implementation exposes the underlying operations:

$$\text{Input} \rightarrow \text{Weighted Sum} \rightarrow \text{Activation} \rightarrow \text{Loss} \rightarrow \text{Gradient} \rightarrow \text{Parameter Update}$$

At the mathematical level, each dense layer computes:

$$\mathbf{z} = W\mathbf{x} + \mathbf{b}, \qquad \mathbf{a} = \sigma(\mathbf{z})$$

Training minimizes $L(\theta)$ by repeatedly applying:

$$\boxed{\theta \leftarrow \theta - \eta \nabla_\theta L}$$

where $\theta$ contains all trainable weights and biases.

---

## Implementation Status

### Implemented

- [x] MNIST IDX parser
- [x] Gzip decompression
- [x] Big-endian binary parsing
- [x] Pixel normalization
- [x] Image flattening
- [x] One-hot encoding
- [x] Neuron implementation
- [x] Dense layers
- [x] Weighted sums
- [x] Biases
- [x] Sigmoid activation
- [x] Sigmoid derivative
- [x] Forward propagation
- [x] MSE loss
- [x] Backpropagation
- [x] Chain-rule gradient computation
- [x] Gradient descent
- [x] Stochastic training
- [x] Digit prediction
- [x] Parameter serialization architecture

### Planned

- [ ] Matrix/vectorized implementation
- [ ] Mini-batch training
- [ ] Softmax output
- [ ] Cross-entropy loss
- [ ] ReLU
- [ ] Xavier initialization
- [ ] He initialization
- [ ] Adam optimizer
- [ ] GPU acceleration
- [ ] Validation pipeline
- [ ] Test-set evaluation
- [ ] Confusion matrix
- [ ] Model checkpoints
- [ ] Training visualizations

---

## Educational Scope

This project connects several areas of computer science and mathematics:

```text
                Neural Network
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   Linear Algebra   Calculus     Optimization
        │             │             │
        ▼             ▼             ▼
     Vectors       Chain Rule   Gradient Descent
     Matrices      Derivatives     Learning
        │             │             │
        └─────────────┼─────────────┘
                      ▼
               Machine Learning
```

At its core, a neural network repeatedly evaluates functions such as $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ and $\mathbf{a} = \sigma(\mathbf{z})$, and computes derivatives such as $\dfrac{\partial L}{\partial \theta}$ to determine how its parameters should change.

The complexity of modern neural networks comes largely from scaling these fundamental operations to millions or billions of parameters, combining them into more sophisticated architectures, and executing them efficiently on specialized hardware.

---

## Mathematical Summary

$$\boxed{\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}}$$

$$\boxed{\mathbf{a}^{(l)} = \sigma\!\left(\mathbf{z}^{(l)}\right)}$$

$$\boxed{\sigma(z) = \frac{1}{1+e^{-z}}}$$

$$\boxed{\sigma'(z) = \sigma(z)(1-\sigma(z))}$$

$$\boxed{L = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}$$

$$\boxed{w \leftarrow w - \eta \frac{\partial L}{\partial w}}$$

$$\boxed{b \leftarrow b - \eta \frac{\partial L}{\partial b}}$$

$$\boxed{\theta_{t+1} = \theta_t - \eta \nabla_\theta L}$$

$$\boxed{\hat{c} = \operatorname*{arg\,max}_i \hat{y}_i}$$

---

## Conclusion

This project implements a feedforward neural network without relying on a machine-learning framework. The complete pipeline is:

```text
Raw MNIST Binary Data
        ↓
IDX Parser
        ↓
Normalized 784-Dimensional Input
        ↓
Dense Layer → Sigmoid → Dense Layer → Sigmoid → Output Layer
        ↓
Loss → Backpropagation → Gradients → Gradient Descent
        ↓
Updated Parameters
```

The implementation provides a low-level view of the relationship between the mathematics of neural networks and their actual software implementation:

$$\boxed{\text{Forward Pass} \rightarrow \text{Loss} \rightarrow \text{Backpropagation} \rightarrow \text{Gradient Descent}}$$

Repeated over many examples and iterations, this process allows the network to learn a mapping from handwritten pixel patterns to digit classes.

---

## Author

**Owolabi Daniel**


Computer science student interested in:

- Computational Complexity
- Machine Learning
- Programming Languages
- Compilers
