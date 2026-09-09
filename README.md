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
- [Input Representation](#input-representation)
- [One-Hot Encoding](#one-hot-encoding)
- [Neuron](#neuron)
- [Dense Layer](#dense-layer)
- [Forward Propagation](#forward-propagation)
- [Activation Function](#activation-function)
- [Loss Function](#loss-function)
- [Backpropagation](#backpropagation)
- [Chain Rule](#chain-rule)
- [Gradient Descent](#gradient-descent)
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [Training Process](#training-process)
- [Prediction](#prediction)
- [Parameter Count](#parameter-count)
- [Computational Complexity](#computational-complexity)
- [Memory Complexity](#memory-complexity)
- [Serialization](#serialization)
- [Running the Project](#running-the-project)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Mathematical Summary](#mathematical-summary)

---

# Architecture

The neural network is a composition of fully connected layers.

For example:

```text
784
 │
 ▼
16
 │
 ▼
16
 │
 ▼
10
```

This corresponds to:

```js
const neural = new NeuralNetwork(784, [16, 16], 10);
```

The architecture is therefore:

```text
Input Layer
784 neurons
    │
    ▼
Hidden Layer
16 neurons
    │
    ▼
Hidden Layer
16 neurons
    │
    ▼
Output Layer
10 neurons
```

The 784 inputs correspond to the pixels of a `28 × 28` MNIST image.

The 10 output neurons correspond to the ten possible digits:

```text
0 1 2 3 4 5 6 7 8 9
```

---

# Project Structure

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

## `parser.js`

The parser is responsible for converting the raw MNIST binary files into JavaScript data structures.

It performs:

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

---

## `Neural Network.js`

This file contains the neural-network implementation.

The implementation is divided conceptually into three levels:

```text
Neuron
   ↓
Layer
   ↓
NeuralNetwork
```

### `Neuron`

Represents an individual computational unit.

### `Layer`

Contains multiple neurons.

### `NeuralNetwork`

Connects multiple layers and performs:

- Forward propagation
- Loss calculation
- Backpropagation
- Parameter updates
- Prediction

---

# MNIST Dataset

The network is trained on the MNIST handwritten digit dataset.

Each example consists of:

```text
28 × 28 grayscale image
+
digit label
```

Therefore every image contains:

$$
28\times28=784
$$

pixel values.

The dataset contains:

```text
60,000 training images
10,000 test images
```

Each image is represented by 784 grayscale pixel intensities.

---

# IDX Binary Format

MNIST is distributed using the IDX binary file format.

The files are gzip compressed:

```text
.gz
 │
 ▼
gzip decompression
 │
 ▼
IDX binary data
 │
 ▼
header parsing
 │
 ▼
images / labels
```

The parser uses Node.js `zlib` to decompress the files.

---

# Label File Format

The label file contains an 8-byte header.

```text
┌────────────────┬────────────────┐
│  Magic Number  │   Item Count   │
│     4 bytes    │     4 bytes    │
└────────────────┴────────────────┘
```

The values are stored using big-endian byte ordering.

They can be read using:

```js
buffer.readUInt32BE()
```

After the header, each label occupies one byte.

Labels can therefore be extracted using:

```js
buffer.readUInt8()
```

---

# Image File Format

The image file contains a 16-byte header:

```text
┌──────────────┬──────────────┬────────┬─────────┐
│ Magic Number │ Image Count  │  Rows  │ Columns │
│   4 bytes    │   4 bytes    │ 4 bytes│ 4 bytes │
└──────────────┴──────────────┴────────┴─────────┘
```

For MNIST:

$$
rows=28
$$

and:

$$
columns=28
$$

Therefore each image contains:

$$
28\times28=784
$$

pixels.

Each pixel occupies one byte.

---

# Input Representation

The raw pixel values are integers:

$$
0\leq p\leq255
$$

The network normalizes them to the interval:

$$
[0,1]
$$

using:

$$
x=\frac{p}{255}
$$

For example:

```text
0   → 0
128 → 0.5019...
255 → 1
```

The two-dimensional image:

```text
28 × 28
```

is flattened into:

```text
784
```

values.

Therefore the input vector is:

$$
\mathbf{x}\in[0,1]^{784}
$$

Conceptually:

```text
28 × 28 image
      │
      ▼
flatten
      │
      ▼
784-dimensional vector
```

---

# One-Hot Encoding

The original MNIST label is an integer:

```text
0 through 9
```

The network represents this as a 10-dimensional vector.

For example, digit `3` becomes:

$$
[0,0,0,1,0,0,0,0,0,0]
$$

Digit `7` becomes:

$$
[0,0,0,0,0,0,0,1,0,0]
$$

In general:

$$
\mathbf{y}\in\{0,1\}^{10}
$$

with exactly one element equal to `1`.

---

# Neuron

The fundamental computational unit of the network is the neuron.

A neuron contains:

- Weights
- Bias
- Activation
- Activation derivative
- Gradient information

Suppose a neuron receives:

$$
\mathbf{x}=
[x_1,x_2,\dots,x_n]
$$

and has weights:

$$
\mathbf{w}=
[w_1,w_2,\dots,w_n]
$$

The neuron first computes its pre-activation:

$$
z=
\sum_{i=1}^{n}w_ix_i+b
$$

Equivalently:

$$
z=\mathbf{w}^{T}\mathbf{x}+b
$$

The result is then passed through an activation function.

---

# Activation Function

This implementation uses the sigmoid activation function.

The sigmoid is:

$$
\sigma(z)=
\frac{1}{1+e^{-z}}
$$

Therefore:

$$
a=\sigma(z)
$$

The sigmoid maps arbitrary real values into:

$$
(0,1)
$$

Its derivative is:

$$
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
$$

Since:

$$
a=\sigma(z)
$$

the derivative can be written as:

$$
\sigma'(z)=a(1-a)
$$

This identity is particularly useful during backpropagation because the neuron already has its activation value.

---

# Dense Layer

A dense layer contains multiple neurons.

Every neuron receives every output from the previous layer.

For example:

```text
                 ┌── Neuron 1
                 │
Input ───────────┼── Neuron 2
                 │
                 ├── Neuron 3
                 │
                 └── Neuron m
```

If the previous layer contains $n$ values and the current layer contains $m$ neurons, the weight matrix has dimensions:

$$
m\times n
$$

The layer can therefore be represented mathematically as:

$$
\mathbf{z}^{(l)}
=
W^{(l)}
\mathbf{a}^{(l-1)}
+
\mathbf{b}^{(l)}
$$

where:

- $W^{(l)}$ is the weight matrix
- $\mathbf{a}^{(l-1)}$ is the previous activation vector
- $\mathbf{b}^{(l)}$ is the bias vector
- $\mathbf{z}^{(l)}$ is the pre-activation vector

The activation is then:

$$
\mathbf{a}^{(l)}
=
\sigma(\mathbf{z}^{(l)})
$$

---

# Forward Propagation

Forward propagation evaluates the network from the input layer toward the output layer.

The first layer receives:

$$
\mathbf{a}^{(0)}=\mathbf{x}
$$

For every subsequent layer:

$$
\mathbf{z}^{(l)}
=
W^{(l)}
\mathbf{a}^{(l-1)}
+
\mathbf{b}^{(l)}
$$

followed by:

$$
\mathbf{a}^{(l)}
=
\sigma(\mathbf{z}^{(l)})
$$

The final activation vector is:

$$
\hat{\mathbf{y}}
=
\mathbf{a}^{(L)}
$$

where $L$ is the output layer.

Conceptually:

```text
Input
  │
  ▼
Weighted Sum
  │
  ▼
Sigmoid
  │
  ▼
Hidden Layer
  │
  ▼
Weighted Sum
  │
  ▼
Sigmoid
  │
  ▼
Output
```

---

# Computational Graph

A single neuron can be represented as:

```text
x₁ ──× w₁──┐
x₂ ──× w₂──┤
x₃ ──× w₃──┤
            ├── Σ ── + b ── sigmoid ──→ a
xₙ ──× wₙ──┘
```

Mathematically:

$$
x_i
\rightarrow
w_ix_i
\rightarrow
\sum_iw_ix_i+b
\rightarrow
\sigma(z)
\rightarrow
a
$$

Forward propagation evaluates this graph from left to right.

Backpropagation propagates gradient information from right to left.

---

# Loss Function

The current implementation uses Mean Squared Error.

For a target vector:

$$
\mathbf{y}
$$

and predicted vector:

$$
\hat{\mathbf{y}}
$$

the loss is:

$$
L=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

For MNIST:

$$
n=10
$$

therefore:

$$
L=
\frac{1}{10}
\sum_{i=1}^{10}
(y_i-\hat{y}_i)^2
$$

The objective of training is to minimize:

$$
L
$$

with respect to the network parameters.

---

# Backpropagation

Backpropagation computes the derivatives of the loss with respect to the network parameters.

For a weight $w_i$:

$$
\frac{\partial L}{\partial w_i}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w_i}
$$

Since:

$$
a=\sigma(z)
$$

we have:

$$
\frac{\partial a}{\partial z}
=
a(1-a)
$$

Since:

$$
z=
\sum_iw_ix_i+b
$$

we have:

$$
\frac{\partial z}{\partial w_i}=x_i
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial w_i}
=
\frac{\partial L}{\partial a}
a(1-a)x_i
}
$$

For the bias:

$$
\frac{\partial z}{\partial b}=1
$$

therefore:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial a}
a(1-a)
}
$$

---

# Chain Rule

The chain rule is the mathematical foundation of backpropagation.

Suppose:

$$
L=L(a)
$$

and:

$$
a=a(z)
$$

and:

$$
z=z(w)
$$

Then:

$$
\frac{dL}{dw}
=
\frac{dL}{da}
\frac{da}{dz}
\frac{dz}{dw}
$$

A neural network contains many such nested functions.

For example:

$$
x
\rightarrow
z^{(1)}
\rightarrow
a^{(1)}
\rightarrow
z^{(2)}
\rightarrow
a^{(2)}
\rightarrow
L
$$

Backpropagation efficiently applies the chain rule through this computational graph.

---

# Backpropagation Through Multiple Layers

Define the error signal:

$$
\delta^{(l)}
=
\frac{\partial L}
{\partial\mathbf{z}^{(l)}}
$$

For the output layer:

$$
\delta^{(L)}
=
\frac{\partial L}
{\partial\mathbf{a}^{(L)}}
\odot
\sigma'(\mathbf{z}^{(L)})
$$

where $\odot$ denotes element-wise multiplication.

For a hidden layer:

$$
\delta^{(l)}
=
\left(
W^{(l+1)T}
\delta^{(l+1)}
\right)
\odot
\sigma'(\mathbf{z}^{(l)})
$$

The weight gradient is:

$$
\frac{\partial L}
{\partial W^{(l)}}
=
\delta^{(l)}
\left(
\mathbf{a}^{(l-1)}
\right)^T
$$

The bias gradient is:

$$
\frac{\partial L}
{\partial\mathbf{b}^{(l)}}
=
\delta^{(l)}
$$

This allows gradient information to propagate backward through an arbitrary number of layers.

---

# Gradient Descent

After calculating the gradients, the parameters are updated.

For a weight:

$$
w
\leftarrow
w-
\eta
\frac{\partial L}{\partial w}
$$

For a bias:

$$
b
\leftarrow
b-
\eta
\frac{\partial L}{\partial b}
$$

where $\eta$ is the learning rate.

In vector form:

$$
\boxed{
\theta_{t+1}
=
\theta_t-
\eta\nabla_\theta L
}
$$

where:

- $\theta$ represents all trainable parameters
- $\nabla_\theta L$ is the gradient
- $\eta$ is the learning rate

---

# Why the Negative Gradient?

The gradient:

$$
\nabla L
$$

points in the direction of greatest local increase of the loss.

Therefore:

$$
-\nabla L
$$

points in the direction of greatest local decrease.

Gradient descent therefore performs:

$$
\theta
\leftarrow
\theta-\eta\nabla L
$$

to move the parameters toward lower-loss regions of the parameter space.

---

# Stochastic Gradient Descent

The current implementation updates the network after processing individual training examples.

Conceptually:

```text
for each epoch:

    for each example:

        forward propagation

        calculate loss

        backpropagate

        update weights

        update biases
```

Because parameters are updated after each example, this is a form of Stochastic Gradient Descent.

For training example $t$:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\nabla_\theta L_t
$$

where $L_t$ is the loss for the current example.

---

# Training Process

The complete training pipeline is:

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
Pixel extraction
       │
       ▼
Normalization
       │
       ▼
One-hot encoding
       │
       ▼
Training example
       │
       ▼
Forward propagation
       │
       ▼
Prediction
       │
       ▼
Loss calculation
       │
       ▼
Backpropagation
       │
       ▼
Gradient calculation
       │
       ▼
Gradient descent
       │
       ▼
Updated parameters
       │
       └───────────────► Next example
```

One epoch consists of one complete pass through the training dataset.

If:

- $N$ = number of examples
- $E$ = number of epochs

then the number of example-level updates is approximately:

$$
N\times E
$$

For the 60,000-example MNIST training set, 20 epochs means:

$$
60,000\times20
=
1,200,000
$$

individual training updates.

---

# Training Example

A network can be initialized as:

```js
const neural = new NeuralNetwork(
    784,
    [16, 16],
    10
);
```

Training:

```js
neural.train(trainingData, 20);
```

A smaller dataset can be used for initial testing:

```js
const smallData = trainingData.slice(0, 1000);

neural.train(smallData, 20);
```

This is useful for verifying the implementation before running a full training process.

---

# Prediction

After training, the network produces ten output activations:

```text
[
    y₀,
    y₁,
    y₂,
    y₃,
    y₄,
    y₅,
    y₆,
    y₇,
    y₈,
    y₉
]
```

The predicted class is:

$$
\hat{c}
=
\operatorname*{arg\,max}_{i}
\hat{y}_i
$$

For example:

```text
[
    0.01,
    0.02,
    0.03,
    0.91,
    0.01,
    0.00,
    0.01,
    0.00,
    0.01,
    0.01
]
```

The maximum activation occurs at index `3`.

Therefore:

```text
prediction = 3
```

A prediction call can have the form:

```js
const result = neural.run(
    example.input,
    example.output
);
```

which can return:

```js
{
    predicted: [3, 0.91],
    correct: true
}
```

---

# Output Activation and Confidence

The highest sigmoid activation can be interpreted as an informal confidence score.

For example:

```text
[
    0.01,
    0.02,
    0.03,
    0.91,
    ...
]
```

would produce:

```text
prediction = 3
confidence ≈ 0.91
```

However, this is **not a calibrated probability**.

Because sigmoid outputs are independent:

$$
\sum_i\hat{y}_i
$$

is not constrained to equal 1.

For a proper multi-class probability distribution, a softmax output layer would be more appropriate:

$$
P(y_i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

which guarantees:

$$
\sum_iP(y_i)=1
$$

---

# Parameter Count

For a fully connected layer connecting $n$ input neurons to $m$ output neurons:

Number of weights:

$$
n\times m
$$

Number of biases:

$$
m
$$

Total parameters:

$$
P=nm+m
$$

For the architecture:

```text
784 → 16 → 16 → 10
```

the parameter counts are:

## Input → Hidden Layer 1

$$
784\times16+16
=
12,560
$$

## Hidden Layer 1 → Hidden Layer 2

$$
16\times16+16
=
272
$$

## Hidden Layer 2 → Output

$$
16\times10+10
=
170
$$

Therefore:

$$
12,560+272+170
=
13,002
$$

The network contains:

$$
\boxed{13,002}
$$

trainable parameters.

---

# Computational Complexity

For a dense layer containing $n$ inputs and $m$ neurons, the forward pass requires approximately:

$$
O(nm)
$$

operations.

For a network containing $L$ layers:

$$
O
\left(
\sum_{l=1}^{L}
n_{l-1}n_l
\right)
$$

For training:

- $N$ = number of training examples
- $E$ = number of epochs

the approximate complexity is:

$$
O
\left(
NE
\sum_{l=1}^{L}
n_{l-1}n_l
\right)
$$

The exact runtime depends on implementation details, JavaScript engine behavior, memory access, and the cost of the activation and gradient calculations.

---

# Memory Complexity

The parameter count is:

$$
P=
\sum_{l=1}^{L}
(n_{l-1}n_l+n_l)
$$

Therefore parameter storage is approximately:

$$
O(P)
$$

For a single training example, intermediate activations require approximately:

$$
O
\left(
\sum_{l=0}^{L}n_l
\right)
$$

additional storage.

Backpropagation requires these intermediate values because the gradients depend on activations generated during the forward pass.

---

# Serialization

The learned parameters can be serialized so that a trained model can be saved and restored.

A serialized model contains the learned:

```text
weights
biases
```

The architecture must remain compatible.

For example:

```text
784 → 16 → 16 → 10
```

cannot directly load parameters belonging to:

```text
784 → 32 → 10
```

because the parameter dimensions are different.

A serialization interface can use methods such as:

```js
toJSON()
```

and:

```js
loadWeights()
```

to save and restore learned parameters.

---

# Running the Project

## Requirements

- Node.js
- ES module support
- MNIST dataset
- `"type": "module"` in `package.json`

Example:

```json
{
    "type": "module"
}
```

The project uses ES module syntax:

```js
import ...
```

and:

```js
export ...
```

---

## Dataset

Place the MNIST files inside:

```text
./data/
```

Expected files:

```text
data/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

---

## Running

Execute:

```bash
node "Neural Network.js"
```

---

# Limitations

## 1. No Vectorization

The implementation performs many operations using ordinary JavaScript arrays and loops.

Modern neural-network implementations usually express the same computations as matrix or tensor operations:

$$
Z=WX+b
$$

This implementation does not currently exploit highly optimized matrix-multiplication kernels.

---

## 2. No GPU Acceleration

The implementation runs using ordinary JavaScript execution.

There is currently no GPU backend.

Consequently, scaling the network to substantially larger architectures would be inefficient.

---

## 3. MSE Instead of Cross-Entropy

The implementation uses Mean Squared Error:

$$
L=
\frac{1}{n}
\sum_i(y_i-\hat{y}_i)^2
$$

For multi-class classification, categorical cross-entropy is generally more appropriate.

A future implementation could use:

$$
L=
-\sum_i y_i\log(\hat{y}_i)
$$

together with softmax.

---

## 4. Sigmoid Throughout the Network

Sigmoid is mathematically convenient and useful for understanding backpropagation.

However, deep neural networks commonly use ReLU-family activations in hidden layers.

For ReLU:

$$
\operatorname{ReLU}(x)
=
\max(0,x)
$$

---

## 5. No Mini-Batch Training

The current implementation updates parameters after individual examples.

Mini-batch training would instead compute gradients over a group of examples:

```text
examples
   │
   ▼
mini-batch
   │
   ▼
forward propagation
   │
   ▼
aggregate gradients
   │
   ▼
parameter update
```

This becomes particularly advantageous when combined with vectorized matrix operations.

---

## 6. Basic SGD Optimizer

The implementation uses basic gradient descent / SGD.

More sophisticated optimizers could include:

- Momentum
- RMSProp
- Adam

---

## 7. No Regularization

The current implementation does not include:

- L1 regularization
- L2 regularization
- Dropout
- Early stopping

These techniques can help control overfitting in larger models.

---

## 8. Limited Evaluation Pipeline

A production machine-learning workflow should distinguish between:

```text
Training Set
Validation Set
Test Set
```

The current project primarily focuses on the training and inference mechanisms.

---

# Future Improvements

## Vectorized Matrix Operations

Replace repeated scalar calculations with matrix operations:

$$
Z=WX+b
$$

and:

$$
A=\sigma(Z)
$$

This would make the implementation substantially more efficient.

---

## Mini-Batch Gradient Descent

Implement mini-batch training:

```text
Dataset
   │
   ├── Batch 1
   ├── Batch 2
   ├── Batch 3
   └── ...
```

Gradients could then be averaged across the batch before updating the parameters.

---

## Softmax Output Layer

Replace the sigmoid output layer with softmax:

$$
P(y_i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

This would produce a normalized probability distribution over the ten digit classes.

---

## Cross-Entropy Loss

Implement categorical cross-entropy:

$$
L=
-\sum_i y_i\log(\hat{y}_i)
$$

For one-hot encoded labels, this reduces to the negative logarithm of the probability assigned to the correct class.

---

## ReLU

Implement:

$$
f(x)=\max(0,x)
$$

for hidden layers.

This would also provide an opportunity to study the difference between sigmoid and ReLU networks.

---

## Xavier / He Initialization

Implement better parameter initialization schemes.

For example, Xavier initialization attempts to maintain approximately stable activation variance across layers.

He initialization is commonly used with ReLU-family activations.

---

## Adam Optimizer

Implement Adam, which combines ideas related to momentum and adaptive learning rates.

The optimizer maintains moving estimates of:

$$
m_t
$$

and:

$$
v_t
$$

and uses them to adapt parameter updates.

---

## Evaluation Metrics

Add:

- Training accuracy
- Validation accuracy
- Test accuracy
- Training loss
- Validation loss
- Confusion matrix
- Per-class accuracy

---

## Model Checkpoints

Automatically save the model whenever validation performance improves.

For example:

```text
Epoch 1  → 89.2%
Epoch 2  → 91.4%
Epoch 3  → 92.1%  ← save
Epoch 4  → 91.8%
```

---

## GPU Acceleration

A future version could use a tensor library or GPU backend for large matrix operations.

The educational implementation would remain useful as the reference implementation against which the optimized implementation could be compared.

---

# Design Philosophy

The project intentionally avoids high-level machine-learning abstractions.

Instead of calling:

```js
model.fit(...)
```

the implementation exposes the underlying operations.

The fundamental training loop can be summarized as:

$$
\text{Input}
\rightarrow
\text{Weighted Sum}
\rightarrow
\text{Activation}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Parameter Update}
$$

At the mathematical level, the core operation of each dense layer is:

$$
\mathbf{z}
=
W\mathbf{x}
+
\mathbf{b}
$$

followed by:

$$
\mathbf{a}
=
\sigma(\mathbf{z})
$$

Training then attempts to minimize:

$$
L(\theta)
$$

by repeatedly applying:

$$
\boxed{
\theta
\leftarrow
\theta-\eta\nabla_\theta L
}
$$

where $\theta$ contains all trainable weights and biases.

---

# Implementation Status

## Implemented

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

## Planned

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

# Educational Scope

This project connects several areas of computer science and mathematics:

```text
                Neural Network
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   Linear Algebra   Calculus   Optimization
        │             │             │
        ▼             ▼             ▼
     Vectors       Chain Rule   Gradient Descent
     Matrices      Derivatives     Learning
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
               Machine Learning
```

The implementation demonstrates how these mathematical ideas combine to produce a trainable computational system.

At its core, a neural network is repeatedly evaluating functions such as:

$$
\mathbf{z}=W\mathbf{x}+\mathbf{b}
$$

$$
\mathbf{a}=\sigma(\mathbf{z})
$$

and computing derivatives such as:

$$
\frac{\partial L}{\partial\theta}
$$

to determine how its parameters should change.

The complexity of modern neural networks comes largely from scaling these fundamental operations to millions or billions of parameters, combining them into more sophisticated architectures, and executing them efficiently on specialized hardware.

---

# Mathematical Summary

For a dense layer:

$$
\boxed{
\mathbf{z}^{(l)}
=
W^{(l)}
\mathbf{a}^{(l-1)}
+
\mathbf{b}^{(l)}
}
$$

Activation:

$$
\boxed{
\mathbf{a}^{(l)}
=
\sigma(\mathbf{z}^{(l)})
}
$$

Sigmoid:

$$
\boxed{
\sigma(z)
=
\frac{1}{1+e^{-z}}
}
$$

Sigmoid derivative:

$$
\boxed{
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
}
$$

MSE:

$$
\boxed{
L=
\frac{1}{n}
\sum_i
(y_i-\hat{y}_i)^2
}
$$

Weight update:

$$
\boxed{
w
\leftarrow
w-
\eta
\frac{\partial L}{\partial w}
}
$$

Bias update:

$$
\boxed{
b
\leftarrow
b-
\eta
\frac{\partial L}{\partial b}
}
$$

General gradient descent:

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta L
}
$$

Prediction:

$$
\boxed{
\hat{c}
=
\operatorname*{arg\,max}_i
\hat{y}_i
}
$$

---

# Conclusion

This project implements a feedforward neural network without relying on a machine-learning framework.

The complete pipeline is:

```text
Raw MNIST Binary Data
        ↓
IDX Parser
        ↓
Normalized 784-Dimensional Input
        ↓
Dense Layer
        ↓
Sigmoid
        ↓
Dense Layer
        ↓
Sigmoid
        ↓
Output Layer
        ↓
Loss
        ↓
Backpropagation
        ↓
Gradients
        ↓
Gradient Descent
        ↓
Updated Parameters
```

The implementation provides a low-level view of the relationship between the mathematics of neural networks and their actual software implementation.

The central idea is simple:

$$
\boxed{
\text{Forward Pass}
\rightarrow
\text{Loss}
\rightarrow
\text{Backpropagation}
\rightarrow
\text{Gradient Descent}
}
$$

Repeated over many examples and iterations, this process allows the network to learn a mapping from handwritten pixel patterns to digit classes.
# Author 
Owolabi Daniel 
Computer science student intrested in.
- Computational Complexity
- Machine learning
- Programming Languages 
- Compilers 