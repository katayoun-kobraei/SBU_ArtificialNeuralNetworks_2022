
# Artificial Neural Networks - Assignment 1  

## Project Overview

This project implements a multi-layer perceptron (MLP) neural network using PyTorch to solve a clothing classification problem using the **Fashion MNIST** dataset. The dataset consists of 10 classes, 60,000 training images, and 10,000 testing images. The project also investigates several machine learning concepts, such as **L1 and L2 regularization**, **dropout**, **batch normalization**, and **early stopping**. The model performance is evaluated through experiments with different hidden layers and regularization techniques.

In addition, a **detailed report** on **EfficientNet**, **ResNext**, and **Inception-ResNet** architectures is included, providing insights into their innovations, advantages, and drawbacks.

---

### Exercise 1: L1 vs L2 Regularization
This exercise involves explaining the differences between L1 and L2 regularization techniques in machine learning models, focusing on how each one affects model weights and sparsity. My answer for this exercise exists in **h3(q1-q2) - 99222084.pdf** file.

---

### Exercise 2: Fashion MNIST Classification with MLP
In this exercise, a multi-layer perceptron (MLP) is implemented using PyTorch to classify images from the Fashion MNIST dataset. Several techniques and architectural decisions are analyzed:

1. **Depth Effect of Hidden Layers**: The effect of varying the number of hidden layers on model performance is reported.
2. **Dropout Technique**: Dropout is used to prevent overfitting, and its results are analyzed.
3. **Early Stopping**: Early stopping is employed to prevent the model from training for too long once the validation loss starts increasing.
4. **Batch Normalization**: Batch normalization is incorporated, and its impact on model training speed and performance is discussed.
5. **Regularization with L1 and L2**: The model is trained using both L1 and L2 regularization, and their results are compared.
6. **Regularization Code**: A code snippet to add a regularization term for the weights is provided.

The MLP is trained and evaluated on the Fashion MNIST dataset, and performance metrics such as training/test loss and accuracy are reported. The report of this exercise also exists in **h3(q1-q2) - 99222084.pdf** file. You can fine the implementation in **q3 (3).ipynb** file.

--- 

## Installation

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- tqdm

### Setup

To install the required libraries, run:

```bash
pip install torch torchvision matplotlib numpy tqdm
```

---

## Data Preparation

This project uses the **Fashion MNIST** dataset, a popular benchmark for image classification tasks. The data is automatically downloaded using `torchvision.datasets.FashionMNIST`. Augmentation and normalization techniques are applied to the dataset using PyTorch’s `transforms` module.

The data preprocessing includes:
- **RandomResizedCrop**
- **Horizontal flipping**
- **Normalization**



## Model Implementation

The core of the project is a **Multi-layer Perceptron (MLP)** implemented using **PyTorch**. The architecture includes:
- Pre-trained **ResNet101** backbone (transfer learning).
- Custom fully connected layers for classification.
- Techniques like **dropout**, **batch normalization**, and **L1/L2 regularization** are applied.

### Model Components:
- **Dropout:** Randomly disables some neurons during training to prevent overfitting.
- **Batch Normalization:** Normalizes activations to improve training speed and stability.
- **L1/L2 Regularization:** Penalizes large weights to simplify the model and reduce overfitting.



## Results & Analysis

The model achieves the following performance:
- **Training Accuracy:** ~94.22%
- **Test Accuracy:** ~89.00%

The impact of different **depths** (number of hidden layers), **dropout**, **batch normalization**, and **regularization** is analyzed and plotted for better visualization.



## References

1. **Fashion MNIST Dataset:** [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
2. **PyTorch Documentation:** [PyTorch](https://pytorch.org/docs/)
3. **EfficientNet Paper:** [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
4. **Inception-ResNet:** [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
5. **ResNeXt:** [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

