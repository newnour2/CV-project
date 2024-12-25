# Sign Language Numbers Classifier

This project implements a comprehensive pipeline to classify hand gestures representing numbers in sign language. The program uses multiple machine learning and deep learning models to train, evaluate, and compare their performances on the dataset. The models used include:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Convolutional Neural Networks (CNN)**

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Models and Methods](#models-and-methods)
- [Results Visualization](#results-visualization)
- [Author](#author)

## Project Structure

- `ModelsClass`: The primary class containing methods for loading data, training models, and visualizing results.
- `load_data`: Loads and preprocesses the dataset for training and testing.
- `train_knn`, `train_logistic_regression`, `train_svm`: Train individual traditional machine learning models.
- `build_neural_network`: Builds and trains a convolutional neural network.
- `visualize_results`: Visualizes training data, learning curves, and model results.
- `plot_confusion_matrix`: Generates confusion matrices for evaluation.
- `run_complete_analysis`: Runs a full pipeline to train, evaluate, and compare all models.

## Features

- Preprocessing:
  - Normalization of image pixel values.
  - Histogram equalization for contrast enhancement.
  - Edge detection and feature extraction.
- Training:
  - Machine learning models (KNN, Logistic Regression, SVM).
  - Deep learning using a CNN architecture.
- Evaluation:
  - Metrics: Accuracy, Precision, Recall, F1-Score.
  - Visualization of confusion matrices.
- Comparison: Compares the performance of all models side-by-side.

## Dataset

The dataset used consists of hand gesture images representing numbers in sign language. Each image corresponds to one of the digits (0-9). The dataset must be formatted as follows:

- **Inputs** (Images): A NumPy array of images where each image is grayscale and normalized between 0 and 1.
- **Labels** (number): A NumPy array of integer labels corresponding to the numbers (0-9).

Ensure you split the dataset into training and testing sets before running the script.

**datasete** link: https://github.com/ardamavi/Sign-Language-Digits-Dataset

## Requirements

Install the necessary libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Models and Methods

1. **K-Nearest Neighbors (KNN)**: Simple and interpretable, used with flat feature vectors.
2. **Logistic Regression**: A linear model for classification, handles multi-class problems.
3. **Support Vector Machine (SVM)**: Classifier using kernel methods for complex boundaries.
4. **Convolutional Neural Network (CNN)**: A deep learning model for image recognition.

## Results Visualization

- **Sample Images**: Displays training samples with their corresponding labels.
- **Confusion Matrices**: Heatmaps showing true vs. predicted labels for each model.
- **Learning Curves**: Plots training and validation accuracy/loss for the neural network.

## Author

Nour Ali Gouda

Mazzen Fazza
