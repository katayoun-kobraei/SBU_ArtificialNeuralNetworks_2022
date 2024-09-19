# Leaky ReLU Implementation for Image Classification

## Overview

This repository contains an implementation of image classification using a modified ResNet-101 model with Leaky ReLU activation. The primary goal of this project is to demonstrate the advantages of Leaky ReLU over standard ReLU, particularly in addressing the dying ReLU problem and improving training speed.

### Advantages of Leaky ReLU:
1. **Solves the Dying ReLU Problem**: Leaky ReLU allows a small, non-zero gradient when the input is negative, preventing neurons from becoming inactive during training.
2. **Faster Training**: Leaky ReLU helps in maintaining a near-zero mean activation, which can speed up convergence compared to ReLU.

## Files in the Repository

- **`h3(q1-q2) - 99222084.pdf`**: Contains theoretical answers to questions 1 and 2 of the assignment.
- **`q3 (3).ipynb`**: Jupyter notebook with the implementation of the model training and evaluation.
- **`گزارش پیاده سازی تمرین سوم.pdf`**: Report detailing the implementation and results of the third exercise.

## Setup

### Prerequisites

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

You can install the necessary packages using pip:

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### Data

The project uses the Intel Image Classification dataset. If you are running this code on Kaggle, the dataset will be available in the specified directories. If running locally, download the dataset and unzip it into the appropriate directories:

```bash
wget https://example.com/intel-image-classification.zip
unzip intel-image-classification.zip
```

Set the data directories in the notebook as follows:

```python
train_data = '/path/to/seg_train/seg_train'
test_data = '/path/to/seg_test/seg_test'
```

## Implementation

The notebook `q3 (3).ipynb` includes the following steps:

1. **Data Preparation**: 
   - Data augmentation and normalization for training and test datasets.
   - Loading datasets using `torchvision.datasets.ImageFolder` and `DataLoader`.

2. **Model Setup**:
   - Use of a pre-trained ResNet-101 model.
   - Modification of the final fully connected layer to include Leaky ReLU activation.

3. **Training**:
   - Training the model using the Adam optimizer and a learning rate scheduler.
   - Monitoring training and test losses.

4. **Evaluation**:
   - Plotting training and test losses.
   - Computing and printing training and test accuracies.

5. **Results**:
   - Accuracy results for training and test datasets.
   - Plot of loss values over epochs.

## Usage

To run the notebook and reproduce the results, follow these steps:

1. Open `q3 (3).ipynb` in Jupyter Notebook or JupyterLab.
2. Execute all cells to perform data preparation, model training, and evaluation.

## Results

The model achieves the following accuracies:

- **Training Accuracy**: 94.22%
- **Test Accuracy**: 89.00%

The loss curves for both training and test datasets are plotted to visualize the training process.

---

If you have any questions or need further assistance, feel free to reach out!


