
# Perceptron and Gradient Descent

## Overview

This project explores the fundamentals of binary classification by implementing learning algorithms from scratch using Python and NumPy. It focuses on understanding how models learn from data, update their parameters, and establish decision boundaries.

## Concepts

### 1. Linear Classification

Linear classification separates data into classes using a linear decision boundary. For an input feature vector \(x\), the model calculates a score:

\[
z = w^T x + b
\]

where:
- \(x\) represents the input features.
- \(w\) represents the weights associated with the features.
- \(b\) is the bias.
- \(z\) is the linear score.

The weights and bias determine the position and orientation of the decision boundary.

### 2. Perceptron Learning Rule

The perceptron is a linear binary classifier that predicts a class using a step-function decision rule.

\[
\hat{y} =
\begin{cases}
1 & z > 0\\
0 & z \leq 0
\end{cases}
\]

The perceptron learning rule updates weights and bias when a sample is misclassified:

\[
w \leftarrow w + \eta(y-\hat{y})x
\]

\[
b \leftarrow b + \eta(y-\hat{y})
\]

Here, \(\eta\) is the learning rate. The updates move the decision boundary to correct classification errors.

For linearly separable data, the perceptron convergence theorem guarantees convergence under the standard perceptron learning rule.

### 3. Sigmoid Activation

The sigmoid function converts a real-valued score into a value between 0 and 1:

\[
\sigma(z) = \frac{1}{1+e^{-z}}
\]

This output can be interpreted as a predicted probability in a model trained using an appropriate probabilistic objective.

Unlike the perceptron, logistic regression uses sigmoid activation to produce probabilities.

### 4. Binary Cross-Entropy Loss

Binary cross-entropy (BCE) measures the difference between true binary labels and predicted probabilities.

\[
L = -\left[y\log(\hat{y})+
(1-y)\log(1-\hat{y})\right]
\]

Lower BCE indicates better probabilistic predictions. The loss penalizes confident predictions that disagree with the true label.

### 5. Gradient Descent

Gradient descent is an optimization algorithm that adjusts model parameters to reduce a loss function.

For a parameter \(w\), the update is:

\[
w \leftarrow w-\eta\frac{\partial L}{\partial w}
\]

The same principle applies to the bias. The gradient indicates the direction of increasing loss, so the parameters are updated in the opposite direction.

For binary classification using sigmoid and BCE, the model can calculate gradients with respect to weights and bias and iteratively minimize the average loss.

### 6. Perceptron Trick vs. Gradient Descent

The perceptron learning rule updates parameters based on classification errors. Gradient descent with BCE updates parameters according to the gradient of a probabilistic loss.

The perceptron focuses on finding a separating decision boundary, while BCE provides information about the quality of predicted probabilities. These approaches therefore use different learning objectives.

## Implementation

The concepts are implemented from scratch using NumPy.

- Weight and bias initialization
- Prediction and parameter updates
- Sigmoid activation and binary cross-entropy
- Gradient calculation and optimization
- Decision boundary visualization using Matplotlib
- Training and test evaluation

The model implementations support an arbitrary number of numerical input features. The current decision-boundary visualization is limited to two features.

## Experimental Results

| Metric | Perceptron Trick | Gradient Descent + BCE |
|---|---:|---:|
| Training BCE | 0.2812 | 0.2295 |
| Test BCE | 0.8380 | 0.2951 |
| Test Accuracy | 65% | 90% |

**Note:** Sigmoid was applied to the perceptron's scores only to calculate BCE for evaluation. The perceptron itself does not use sigmoid or optimize BCE.

## Conclusion

This experiment demonstrates two different approaches to learning a linear decision boundary. The perceptron uses classification-error-based updates, while gradient descent with binary cross-entropy optimizes a differentiable probabilistic loss.

On the current dataset and test split, gradient descent achieved higher test accuracy and lower test BCE. These results describe this particular experiment and do not establish that one approach will always perform better.

## Technologies

- Python
- NumPy
- Matplotlib
- Jupyter Notebook
