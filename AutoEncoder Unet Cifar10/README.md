# Artificial Neural Networks - 4th Assignment

This repository contains the theoretical answers and practical implementation for the fourth assignment in the **Artificial Neural Networks** course at Shahid Beheshti University (Bachelor’s Program). The assignment is divided into three exercises, with theory questions on Variational Autoencoders (VAE) and an implementation task using CIFAR-10 images.

### Repository Contents

- **`Exercise 1-2.pdf`**: Answers to the theory questions from Exercise 1 and 2.
- **`homework4 (1).ipynb`**: Jupyter notebook with the implementation for Exercise 3.
- **`گزارش کار پیاده سازی بخش سوم.pdf`**: Report of the implementation for Exercise 3 (in Persian).
- **`implementation report.pdf`**: Report of the implementation for Exercise 3 (in English).

---

## Exercises Overview

### Exercise 1

In this exercise, you will explore how the architecture of a **Variational Autoencoder (VAE)** enables it to generate new data points, and how it differs from a traditional **Associative Autoencoder** that cannot perform this task. The answer is provided in the `Exercise 1-2.pdf` document.

### Exercise 2

This exercise focuses on the variational autoencoder's optimization process, particularly the lower bound of the data likelihood for an input sample \( x^{(i)} \). Specifically, the following questions are addressed:

- The role of the **KL-divergence** term in the optimization.
- The advantage of modeling \( p_\theta(z) \) and \( q_\phi(z|x^{(i)}) \) using normal distributions with diagonal covariance matrices.
- The task of the first term in the equation and its effect on the latent space.

Answers to these theoretical questions are also included in the `Exercise 1-2.pdf` document.

### Exercise 3

This practical task requires creating a neural network using the **CIFAR-10** dataset. The network should take two images \( x_1 \) and \( x_2 \) as input, and learn to reconstruct both images when only given their average \( \frac{x_1 + x_2}{2} \) as input. The architecture and design are open-ended, allowing for creative solutions.

- **Extra Challenge**: An additional +20 points are awarded to the student who achieves the lowest loss on the test data.

The implementation of this exercise is found in the `homework4 (1).ipynb` file, and the details of the implementation are discussed in both Persian (`گزارش کار پیاده سازی بخش سوم.pdf`) and English (`implementation report.pdf`) reports.

---

## Setup Instructions

### Prerequisites

To run the notebook and reproduce the results, you will need the following libraries:

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

You can install the required libraries using pip:

```bash
pip install torch torchvision numpy matplotlib
```

### Running the Code

1. Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Open the notebook `homework4 (1).ipynb` in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook homework4 (1).ipynb
```

3. Follow the steps in the notebook to:

   - Load the CIFAR-10 dataset.
   - Build and train the network to generate both images from their average.
   - Evaluate the performance of the network and compute the loss.

### Dataset

The CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, is used for the practical part of this assignment. The dataset can be loaded directly using PyTorch's `torchvision.datasets` module within the notebook.

---

## Results

The results, including the training loss, test loss, and reconstructed images, are presented in the notebook. The detailed reports (`گزارش کار پیاده سازی بخش سوم.pdf` and `implementation report.pdf`) provide further insights into the network architecture, training process, and experimental outcomes.

---

Feel free to reach out if you have any questions regarding this assignment!

---

