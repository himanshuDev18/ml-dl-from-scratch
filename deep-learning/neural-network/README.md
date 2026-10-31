# General Feedforward Neural Network for Binary Classification using Batch Gradient Descent from Scratch

A general feedforward neural network implemented from scratch using **NumPy** for binary classification.

The network supports a configurable number of hidden layers and neurons. Forward propagation, backpropagation, and parameter updates are implemented dynamically using matrix operations and loops.

The current implementation uses:

- **ReLU** for hidden layers
- **Sigmoid** for the output layer
- **Binary Cross-Entropy** as the loss function
- **Batch Gradient Descent** for training

The implementation is tested on the **Make Moons dataset** to verify that the network can learn a nonlinear binary classification problem.

---

# Features

- Configurable number of hidden layers and neurons
- Matrix-based implementation using NumPy
- Dynamic parameter initialization
- Forward propagation
- ReLU and Sigmoid activation functions
- Binary Cross-Entropy loss
- Backpropagation using the chain rule
- Batch Gradient Descent
- Training and test evaluation
- Loss visualization
- Decision boundary visualization

---

# General Network Architecture

The network architecture is defined using a list containing the number of neurons in each layer.

```python
layer_sizes = [2, 8, 4, 1]
```

This creates:

```text
2 input features → 8 neurons → 4 neurons → 1 output neuron
```

However, the implementation is not limited to this architecture.

For example:

```python
[2, 5, 1]
[2, 10, 8, 1]
[2, 16, 8, 4, 1]
```

can use the same implementation.

The first value represents the number of input features, the middle values represent hidden layers, and the final value represents the output layer.

The general structure is:

```text
Input Layer
     ↓
Hidden Layer(s)
     ↓
Output Layer
     ↓
Binary Classification
```

---

# Mathematical Notation

| Symbol | Meaning |
|---|---|
| $N$ | Number of samples |
| $L$ | Number of trainable layers |
| $X$ | Input data |
| $Y$ | True labels |
| $W^{[l]}$ | Weight matrix of layer $l$ |
| $b^{[l]}$ | Bias vector of layer $l$ |
| $Z^{[l]}$ | Linear output of layer $l$ |
| $A^{[l]}$ | Activation output of layer $l$ |
| $\hat{Y}$ | Final prediction |

The input is represented as:

$$
A^{[0]} = X
$$

The implementation uses **samples as rows and features/neurons as columns**.

---

# General Matrix Representation

For a layer $l$:

$$
Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]}
$$

$$
A^{[l]} = g(Z^{[l]})
$$

where $g$ is the activation function.

Suppose:

- $N$ = number of samples
- $n_{l-1}$ = neurons in the previous layer
- $n_l$ = neurons in the current layer

Then:

$$
A^{[l-1]} \in \mathbb{R}^{N \times n_{l-1}}
$$

$$
W^{[l]} \in \mathbb{R}^{n_{l-1} \times n_l}
$$

$$
b^{[l]} \in \mathbb{R}^{1 \times n_l}
$$

Therefore:

```text
(N, nₗ₋₁) @ (nₗ₋₁, nₗ) = (N, nₗ)
```

The bias has shape:

```text
(1, nₗ)
```

NumPy automatically broadcasts the bias across all $N$ samples.

---

# Parameter Initialization

Parameters are created dynamically based on `layer_sizes`.

For every layer:

$$
W^{[l]} \in \mathbb{R}^{n_{l-1} \times n_l}
$$

$$
b^{[l]} \in \mathbb{R}^{1 \times n_l}
$$

For example:

```python
layer_sizes = [2, 8, 4, 1]
```

The parameter shapes are:

```text
W1: (2, 8)
b1: (1, 8)

W2: (8, 4)
b2: (1, 4)

W3: (4, 1)
b3: (1, 1)
```

---

# Activation Functions

## ReLU

ReLU is used in the hidden layers:

$$
ReLU(z) = \max(0, z)
$$

Its derivative is:

$$
ReLU'(z) =
\begin{cases}
1, & z > 0 \\
0, & z \leq 0
\end{cases}
$$

---

## Sigmoid

The output layer uses Sigmoid:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

It produces values between $0$ and $1$, representing probabilities for binary classification.

A threshold of `0.5` converts probabilities into class predictions.

---

# Forward Propagation

For each layer:

$$
Z^{[l]} =
A^{[l-1]}W^{[l]} + b^{[l]}
$$

For hidden layers:

$$
A^{[l]} = ReLU(Z^{[l]})
$$

For the output layer:

$$
\hat{Y} = A^{[L]} = \sigma(Z^{[L]})
$$

The layers are processed dynamically using a loop.

```text
Input X
   ↓
Linear Transformation
   ↓
ReLU
   ↓
Hidden Layer(s)
   ↓
Linear Transformation
   ↓
Sigmoid
   ↓
Prediction
```

---

# Binary Cross-Entropy Loss

The network uses Binary Cross-Entropy loss:

$$
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\left(
y_i \log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right)
$$

where:

- $y_i$ is the actual label
- $\hat{y}_i$ is the predicted probability
- $N$ is the number of samples

The loss is averaged across all samples.

---

# Backpropagation

Backpropagation calculates the gradients of the loss with respect to every weight and bias.

The calculation starts from the output layer and moves backward through the network using the chain rule.

## Output Layer

For Sigmoid activation combined with Binary Cross-Entropy loss:

$$
dZ^{[L]}
=
\frac{\hat{Y} - Y}{N}
$$

The output weight gradient is:

$$
dW^{[L]}
=
(A^{[L-1]})^T dZ^{[L]}
$$

The bias gradient is:

$$
db^{[L]}
=
\sum dZ^{[L]}
$$

The gradient passed to the previous layer is:

$$
dA^{[L-1]}
=
dZ^{[L]}(W^{[L]})^T
$$

---

## Hidden Layers

For hidden layers:

$$
dZ^{[l]}
=
dA^{[l]}
\odot ReLU'(Z^{[l]})
$$

where $\odot$ represents element-wise multiplication.

The weight gradient is:

$$
dW^{[l]}
=
(A^{[l-1]})^T dZ^{[l]}
$$

The bias gradient is:

$$
db^{[l]}
=
\sum dZ^{[l]}
$$

The gradient passed to the previous layer is:

$$
dA^{[l-1]}
=
dZ^{[l]}(W^{[l]})^T
$$

This process is repeated backward through all hidden layers.

---

# Gradient Dimensions

For layer $l$:

```text
A[l-1] : (N, n[l-1])
W[l]   : (n[l-1], n[l])
b[l]   : (1, n[l])

Z[l]   : (N, n[l])
A[l]   : (N, n[l])
```

During backpropagation:

```text
dZ[l]  : (N, n[l])

A[l-1].T : (n[l-1], N)

dW[l]  : (n[l-1], n[l])

db[l]  : (1, n[l])

dA[l-1] : (N, n[l-1])
```

---

# Batch Gradient Descent

This project uses **Batch Gradient Descent**.

During one epoch:

1. All training samples are passed through the network.
2. Predictions are calculated.
3. Binary Cross-Entropy loss is calculated.
4. Gradients are calculated using all training samples.
5. Parameters are updated once.

The parameter update rule is:

$$
W^{[l]}
\leftarrow
W^{[l]}
-
\alpha dW^{[l]}
$$

$$
b^{[l]}
\leftarrow
b^{[l]}
-
\alpha db^{[l]}
$$

where $\alpha$ is the learning rate.

The complete process is:

```text
All Training Samples
        ↓
Forward Propagation
        ↓
Calculate BCE Loss
        ↓
Backpropagation
        ↓
Calculate Gradients
        ↓
Update Parameters
        ↓
Next Epoch
```

---

# Testing the Implementation

After implementing the general neural network, it was tested on the **Make Moons dataset**.

The purpose of this experiment was to check whether the implementation could learn a nonlinear binary classification problem.

The Make Moons dataset contains two classes arranged in a curved pattern, making it useful for testing a neural network.

---

# Dataset

The dataset contains:

```text
500 samples
2 input features
2 classes
```

The labels are reshaped into a column vector:

```python
y = y.reshape(-1, 1)
```

Final shapes:

```text
X: (500, 2)
y: (500, 1)
```

Scikit-learn is used only for:

- Generating the Make Moons dataset
- Splitting the dataset into training and test sets

The neural network itself is implemented manually using NumPy.

---

# Train-Test Split

The dataset is divided into:

```text
80% → Training Data
20% → Test Data
```

The network is trained using only the training data.

The test data is used to evaluate performance on unseen samples.

---

# Experiment Architecture

For the Make Moons experiment:

```python
layer_sizes = [2, 8, 4, 1]
```

Architecture:

```text
2 Input Features
        ↓
8 Hidden Neurons
        ↓
4 Hidden Neurons
        ↓
1 Output Neuron
```

This is only the configuration used for testing. The general implementation can use other layer configurations.

---

# Training Configuration

| Parameter | Value |
|---|---|
| Dataset | Make Moons |
| Samples | 500 |
| Architecture | 2 → 8 → 4 → 1 |
| Hidden Activation | ReLU |
| Output Activation | Sigmoid |
| Loss Function | Binary Cross-Entropy |
| Optimizer | Batch Gradient Descent |
| Learning Rate | 0.5 |
| Epochs | 1000 |
| Test Size | 20% |

---

# Results

The final experiment achieved:

| Metric | Result |
|---|---:|
| Training Accuracy | 98.4% |
| Test Accuracy | 99% |
| Final Training Loss | 0.036 |

The model successfully learned the nonlinear pattern in the Make Moons dataset.

---

# Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn

NumPy is used for all neural network operations.

Matplotlib is used for visualizing the dataset, training loss, and decision boundary.

Scikit-learn is only used for dataset generation and train-test splitting.

---

# Project Structure

```text
neural-network-from-scratch/
│
├── neural_network_from_scratch.ipynb
└── README.md
```

---

# How to Run

Clone the repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

Open the notebook:

```text
neural_network_from_scratch.ipynb
```

Run all cells in order.

---

# What I Learned

Through this project, I implemented and understood:

- Matrix representation of neural networks
- Configurable network architectures
- Weight and bias initialization
- Forward propagation
- ReLU and Sigmoid activation functions
- Binary Cross-Entropy loss
- Backpropagation
- Gradient calculation using the chain rule
- Batch Gradient Descent
- Training and testing a neural network

---

# Future Improvements

Possible extensions include:

- Mini-batch Gradient Descent
- Softmax for multiclass classification
- Multiclass Cross-Entropy loss
- Different activation functions
- Regularization
- Better weight initialization
- A reusable `NeuralNetwork` class
- Saving and loading trained parameters

---

# Note

The main focus of this project is understanding how a general feedforward neural network for binary classification works internally.

The network architecture is configurable, while the current implementation uses ReLU hidden layers, a Sigmoid output layer, Binary Cross-Entropy loss, and Batch Gradient Descent.

The Make Moons dataset is used to test the implementation on a nonlinear binary classification problem.