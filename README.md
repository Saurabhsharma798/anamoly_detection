# Baseline Model Training Documentation

This document explains the baseline anomaly detection models used in the project and describes what each code block in the model training phase does.

The models covered in this phase are:
- Isolation Forest
- One-Class SVM
- Autoencoder

The goal of this phase is to establish reliable baseline performance before moving to ensemble and cross-dataset analysis.

---

## Common Setup

### Libraries Used
- NumPy: numerical operations
- Scikit-learn: classical anomaly detection models and evaluation metrics
- TensorFlow / Keras: autoencoder implementation

### Evaluation Metrics
All models are evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

These metrics are chosen because the dataset is highly imbalanced, making accuracy unreliable.

---

## Evaluation Function

A helper function is defined to compute all evaluation metrics consistently for every model.

Purpose:
- Avoids repeated code
- Ensures fair comparison across models
- Centralizes metric computation

The function takes:
- True labels
- Predicted labels
- Anomaly scores

And returns a dictionary of performance metrics.

---

## Isolation Forest

### Model Initialization
Isolation Forest is initialized with a fixed number of trees and a contamination value close to the expected fraud ratio.

Key parameters:
- `n_estimators`: number of trees in the ensemble
- `contamination`: expected proportion of anomalies
- `max_samples`: controls randomness for better generalization
- `n_jobs = -1`: enables parallel processing using all CPU cores

### Training
The model is trained on the full training dataset.  
Isolation Forest is unsupervised and does not require labels during training.

### Prediction
- Anomaly scores are obtained using the decision function
- Samples predicted as `-1` are treated as anomalies
- Predictions are converted into binary labels

### Purpose
Isolation Forest provides a conservative baseline with low false positives and strong ranking ability.

---

## One-Class SVM

### Model Initialization
One-Class SVM is initialized with an RBF kernel to learn a non-linear boundary around normal data.

Key parameters:
- `kernel`: radial basis function
- `gamma`: controls boundary flexibility
- `nu`: upper bound on the fraction of anomalies

### Training
The model is trained **only on normal data**.

Reason:
- One-Class SVM assumes a single-class distribution
- Including anomalies during training would distort the learned boundary

### Prediction
- Distance from the decision boundary is used as an anomaly score
- Points outside the boundary are labeled as anomalies

### Purpose
One-Class SVM acts as a high-recall model that detects most anomalies but may produce more false positives.

---

## Autoencoder

### Model Architecture
The autoencoder is a neural network with:
- An encoder that compresses input features
- A decoder that reconstructs the original input

The network is symmetric and uses:
- ReLU activations in hidden layers
- Linear activation in the output layer
- Mean Squared Error as the loss function

### Training
The autoencoder is trained **only on normal data**.

Reason:
- The model learns to reconstruct normal transaction patterns
- Anomalous transactions produce higher reconstruction errors

Early stopping is used to prevent overfitting.

### Anomaly Detection
- Reconstruction error is calculated for each test sample
- A percentile-based threshold is applied
- Samples with error above the threshold are classified as anomalies

### Purpose
The autoencoder provides the best balance between precision and reca
