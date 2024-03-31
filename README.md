# Diabetes Prediction ML Project

This project aims to predict whether a person is diabetic or not based on certain health indicators using machine learning techniques.

## Overview

In this project, we utilize the diabetes dataset containing various health indicators such as glucose level, blood pressure, BMI, etc., along with information about whether the person is diabetic or not. We preprocess the data, train a Support Vector Machine (SVM) classifier, and evaluate its performance.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- scikit-learn

## Steps Involved

### 1. Importing the Dependencies

We import necessary libraries such as NumPy, Pandas, and scikit-learn modules for data processing, model training, and evaluation.

### 2. Data Collection and Analysis

We load the diabetes dataset into a Pandas DataFrame and analyze its structure, statistical measures, and class distribution.

### 3. Data Preprocessing

We standardize the features using StandardScaler to ensure uniformity in data distribution, which is a common preprocessing step in machine learning workflows.

### 4. Train-Test Split

We split the dataset into training and testing sets to train the model on one subset and evaluate its performance on another independent subset.

### 5. Model Training

We train a Support Vector Machine (SVM) classifier with a linear kernel using the training data.

### 6. Model Evaluation

We evaluate the trained model's performance on both training and testing datasets using accuracy as the evaluation metric.

### 7. Making Predictions

We demonstrate making predictions on new input data using the trained model and provide a simple predictive system.

## Usage

To run the project:

1. Ensure you have all the dependencies installed.
2. Download the provided `diabetes.csv` dataset.
3. Run the provided Python script containing the code.

## Results

The trained SVM classifier achieves an accuracy score of approximately 78.66% on the training data and 77.27% on the test data.

## Predictive System

We showcase how to use the trained model to make predictions on new input data, indicating whether the person is diabetic or not.

Swastik Bharti
