
## Purpose of This Notebook
This notebook is used to **train and evaluate base (single learner) anomaly detection models**.  
These models provide **baseline performance** that will later be compared with ensemble and GenAI-based approaches.

The notebook focuses only on:
- Training models
- Making predictions
- Evaluating results

---

## Step 1: Importing Libraries
The notebook starts by importing required libraries for:
- Data handling (`numpy`, `pandas`)
- Anomaly detection models
- Performance evaluation metrics

These libraries are necessary to build, train, and evaluate the models.

---

## Step 2: Data and Label Preparation
The dataset used in the notebook is already preprocessed.

Labels are converted into a **binary format**:
- `0` → Normal data
- `1` → Anomalous data (Attack / Fraud)

This conversion ensures compatibility with machine learning evaluation metrics.

---

## Step 3: Evaluation Function
A common evaluation function is defined to calculate:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (if anomaly scores are available)

Using a single function ensures **consistent evaluation** for all models.

---

## Step 4: Isolation Forest Model Training
Isolation Forest is trained as the first baseline model.

- The model learns patterns of normal data.
- Predictions are generated for test data.
- Model outputs are converted into binary anomaly labels.
- Anomaly scores are extracted for ROC-AUC calculation.

---

## Step 5: Isolation Forest Evaluation
The Isolation Forest model is evaluated using:
- Standard classification metrics
- Confusion matrix

This establishes the **initial baseline performance**.

---

## Step 6: One-Class SVM Model Training
One-Class SVM is trained as the second baseline model.

- It learns a boundary around normal data.
- Data points outside this boundary are marked as anomalies.
- Predictions are converted into binary labels.
- Decision scores are used for ROC-AUC.

---

## Step 7: One-Class SVM Evaluation
The One-Class SVM model is evaluated using the same metrics as Isolation Forest.

This allows a **fair comparison** between different base models.

---

## Step 8: Autoencoder Model Definition
An autoencoder neural network is defined with:
- Encoder layers to compress data
- Decoder layers to reconstruct data

The model is designed to learn how normal data looks.

---

## Step 9: Autoencoder Training
The autoencoder is trained using:
- Normal data only
- Reconstruction loss (Mean Squared Error)

The model learns to reconstruct normal patterns accurately.

---

## Step 10: Autoencoder-Based Anomaly Detection
Anomalies are detected using reconstruction error:
- High reconstruction error indicates abnormal behavior
- A percentile-based threshold is used to classify anomalies

This provides a reconstruction-based anomaly detection baseline.

---

## Step 11: Autoencoder Evaluation
The autoencoder predictions are evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

---

## Step 12: Saving Baseline Results
The performance results of all models are stored together.

These results will be used later for:
- Ensemble model construction
- Performance comparison
- Statistical analysis

---

## Summary
This notebook:
- Trains three base anomaly detection models
- Evaluates each model using consistent metrics
- Establishes baseline performance for anomaly detection

The results from this notebook serve as a **reference point** for further improvements using ensemble and GenAI-based methods.
