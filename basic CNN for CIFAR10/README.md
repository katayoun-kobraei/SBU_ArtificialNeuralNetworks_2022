# Artificial Neural Networks - 2nd Assignment

## Overview

This repository contains the second assignment for the Artificial Neural Networks course at Shahid Beheshti University. The assignment includes theoretical questions and practical implementation tasks related to Convolutional Neural Networks (CNNs) using the CIFAR-10 dataset.


### Exercises

1. **Backpropagation in Convolutional Layers**
   - Describe the backpropagation process in convolutional layers. [Reference Link](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)

2. **Symmetry Breaking Phenomenon**
   - What is symmetry breaking in artificial neural networks? How can it be prevented?

3. **Pooling Layers**
   - Discuss the benefits and drawbacks of pooling layers. Are you willing to use these layers frequently?

4. **Cross-Entropy vs. Quadratic Cost**
   - Explain the differences between cross-entropy and quadratic cost functions. Which is preferable for classification problems?

5. **Leaky ReLU vs. ReLU**
   - Compare Leaky ReLU with standard ReLU. Which one is faster? Which one prevents gradient vanishing better?

6. **CNN Implementation on CIFAR-10 Dataset**
   - Implement a convolutional neural network to classify images in the CIFAR-10 dataset. Report on:
     - The effect of different hidden layer depths.
     - The impact of batch normalization on convolutional layers.
     - Optimal model architectures for achieving maximum accuracy.
     - Confusion matrix analysis for your model.

## Files in Repository

- **Homework2 .ipynb:** Jupyter notebook containing the implementation of a CNN for the CIFAR-10 dataset for exercise 6.
- **Homework2.ipynb - Colaboratory.pdf:** PDF version of the Jupyter notebook implementation.
- **Report of question 6.pdf:** Detailed report on the implementation of question 6.
- **homework 2(q1..q5) - 99222084.pdf:** Theoretical answers to questions 1 to 5.

## Implementation Details

### 1. Convolutional Neural Network Implementation
- The CNN is implemented using TensorFlow and Keras.
- Initial model architecture includes several convolutional and max pooling layers.
- Various experiments are conducted to evaluate the effect of layer depth, batch normalization, and dropout.
- Early stopping is used to prevent overfitting and optimize the model.

### 2. Model Training and Evaluation
- **Initial CNN Model:** A basic CNN model with 3 convolutional layers and max pooling.
- **Extended CNN Model:** Added more hidden layers and Global Average Pooling to assess the depth effect.
- **Early Stopping:** Applied to achieve better accuracy and reduce overfitting.
- **Batch Normalization and Dropout:** Included in a revised model to analyze their effects on performance.
- **Confusion Matrix and Classification Report:** Generated to evaluate model performance and accuracy.

### 3. Results
- The models are trained on the CIFAR-10 dataset and evaluated using accuracy, loss, and confusion matrix.
- Detailed analysis of model performance is provided in the `Report of question 6.pdf`.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install required packages:
   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn
   ```

3. Open `Homework2 .ipynb` in Jupyter Notebook or Google Colab.

4. Run the notebook to execute the CNN implementation and analysis.

## Contact

For any questions or issues, please contact [Instructor's Contact Information] or reach out in the course Telegram group.
