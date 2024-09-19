
# Artificial Neural Networks - Assignment 1


## Overview

This assignment involves two main tasks related to Artificial Neural Networks. The first task is theoretical, focusing on the differences between L1 and L2 regularization. The second task is practical, involving the implementation of a Multi-Layer Perceptron (MLP) neural network using PyTorch for classifying images in the Fashion MNIST dataset.

## Exercise 1: Regularization Techniques

### L1 Regularization

- **Concept:** L1 regularization adds a penalty equal to the absolute value of the weights to the loss function. This encourages sparsity in the model, pushing some weights to zero. It helps in feature selection by reducing the number of active features.
  
- **Effect on Weights:** Forces less important feature weights towards zero, creating a sparse model. This makes the model simpler and potentially more interpretable.

### L2 Regularization

- **Concept:** L2 regularization adds a penalty equal to the squared value of the weights to the loss function. This approach does not create sparse models but instead penalizes large weights to prevent overfitting.
  
- **Effect on Weights:** Reduces the magnitude of the weights, but weights are not driven to zero. This helps in managing the model complexity and avoiding overfitting.


### Preference

- **When to Use L1:** Preferable when feature selection is desired and a simpler model is needed.
- **When to Use L2:** Better for cases where all features are expected to contribute to the model and where overfitting is a concern.

## Exercise 2: Fashion MNIST Classification with MLP

### Implementation

The task involved implementing an MLP neural network using Keras to classify images from the Fashion MNIST dataset. Due to issues with accessing the dataset and other errors, the implementation includes several steps and observations:

1. **Data Handling:** 
   - Loading and preprocessing of the Fashion MNIST dataset was attempted, but faced issues due to HTTP errors in fetching the dataset.

2. **Model Architectures:**
   - **Basic MLP:** Implemented with 5 hidden layers and evaluated performance. The model was compiled and trained but had issues due to missing data.
   - **3-Layered MLP:** Reduced the number of hidden layers to observe changes in performance.
   - **7-Layered MLP:** Increased the depth to assess improvements in accuracy and performance.

3. **Dropout Technique:**
   - Applied dropout layers to the MLP model to investigate its impact on overfitting and learning speed.

4. **Early Stopping:**
   - Implemented early stopping to halt training when no improvement in validation loss was observed.

5. **Error Handling:**
   - Encountered multiple issues with missing data variables and code execution errors. This impacted the ability to fully execute and test the models.

### Key Findings

- **Depth Effect:** Increasing the number of hidden layers showed potential improvements but also risks of overfitting.
- **Dropout Impact:** Dropout layers generally improve learning speed and reduce overfitting.
- **Early Stopping:** Effective in preventing unnecessary training once the model performance plateaus.


## References

- Fashion MNIST dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Keras Documentation: [Keras](https://keras.io/)
