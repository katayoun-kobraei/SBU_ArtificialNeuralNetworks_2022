# Artificial Neural Network - Final Presentation
# Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction, and Classification

## Project Overview

This repository contains the implementation of **Generative PointNet**, a novel approach for learning deep energy-based models (EBMs) on 3D point clouds and its presentation slides. This project is based on the paper:

- **Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction, and Classification** [arXiv:2004.01301](https://arxiv.org/abs/2004.01301)

The model presented in this paper offers an innovative energy-based approach to learning generative models on unordered 3D point sets, enabling tasks such as generation, reconstruction, and classification of point clouds. The energy-based model provides explicit density estimation and is trained via Maximum Likelihood Estimation (MLE) using MCMC-based Langevin dynamics for sampling.

### Key Features of the Model:
1. **Energy-Based Modeling**: The discriminator acts as an energy function, giving low energy to data near the real data manifold and high energy to unrealistic data.
2. **3D Point Cloud Processing**: The model handles 3D point clouds, unordered collections of points in space, providing tools for tasks like classification, segmentation, and reconstruction.
3. **Langevin Dynamics Sampling**: The model generates 3D point clouds using MCMC-based Langevin dynamics for sampling.
4. **Applications**: This model performs competitively for 3D point cloud generation, reconstruction, and classification tasks compared to other state-of-the-art methods.
   
## Project Contents

- **`2004.01301.pdf`**: The original paper detailing the Generative PointNet model.
- **`PointNet GAN.pdf`**: Presentation slides summarizing key concepts of GANs and the Generative PointNet model.
- **`deep-energy-based-model.ipynb`**: The implementation of the Generative PointNet model, including training and testing code for point cloud generation, reconstruction, and classification.
- **`گزارش پیاده سازی.pdf`**: A detailed report (in Persian) documenting the implementation, methodology, and results of the Generative PointNet model.

## Files

### 1. `deep-energy-based-model.ipynb`
The main Jupyter notebook contains the implementation of the Generative PointNet model. It includes:
- Model architecture for the energy-based point cloud generator.
- Maximum Likelihood Estimation (MLE) for training the model.
- Langevin Dynamics sampling for generating point clouds.
- Evaluation metrics for reconstruction and classification tasks.

### 2. `PointNet GAN.pdf`
A slide presentation covering:
- An introduction to Generative Adversarial Networks (GANs).
- Point cloud data and its applications.
- Energy-based learning.
- Details of the Generative PointNet model, including experiments and results for 3D point cloud generation, reconstruction, and classification.

### 3. `گزارش پیاده سازی.pdf`
### 3. `A comprehensive report (in Persian and English) documenting`
- The technical details of the implementation.
- The design choices made in building the Generative PointNet model.
- Results and evaluations of the model on different point cloud datasets.

### 4. `2004.01301.pdf`
The paper explaining the theory behind the Generative PointNet model, its architecture, and its applications in point cloud generation, reconstruction, and classification. The paper includes theoretical details such as energy-based modeling, Langevin dynamics, and the experimental results obtained on datasets like ModelNet10.

## Installation and Dependencies

### Requirements

- Python 3.x
- Jupyter Notebook
- NumPy
- PyTorch
- Matplotlib
- Scikit-learn

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/katayoun-kobraei/Generative-PointNet.git
   cd Generative-PointNet
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook for the implementation:
   ```bash
   jupyter notebook deep-energy-based-model.ipynb
   ```

## Running the Code

1. **Training the Model**: 
   In the Jupyter notebook, you can run the code blocks to train the Generative PointNet model on point cloud datasets like ModelNet10.

2. **Generating Point Clouds**: 
   The notebook includes code for sampling 3D point clouds using Langevin dynamics.

3. **Reconstruction and Classification**: 
   The model's performance on reconstruction and classification tasks is also evaluated, and the notebook contains visualizations of the results.

## Results

### Point Cloud Generation
The Generative PointNet model is able to generate realistic 3D point clouds from noise using Langevin dynamics sampling.

### Point Cloud Reconstruction
The model reconstructs 3D point clouds with high accuracy by minimizing the reconstruction error between the generated and actual point clouds.

### Classification
The model achieves competitive performance in point cloud classification tasks, even when the input point clouds are partially observed or corrupted with noise.

## Conclusion

Generative PointNet demonstrates the power of energy-based models for 3D point cloud processing. By unifying the tasks of generation, reconstruction, and classification into a single framework, the model offers a versatile tool for working with 3D point cloud data.

## References

- [Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for 3D Generation, Reconstruction, and Classification (arXiv link)](https://arxiv.org/abs/2004.01301)
- [GAN Lab - Visualize GANs](https://poloclub.github.io/ganlab/)
```
