# Base Model Training – Anomaly Detection Notebook

## Purpose of This Notebook
This notebook is used to **train and evaluate baseline (single learner) anomaly detection models**.  
These models act as a **reference point** for later comparison with ensemble and GenAI-based methods.

The notebook focuses only on **model training and evaluation**, not ensembling or data augmentation.

---

## Step 1: Import Required Libraries
In this step, all necessary Python libraries are imported.

- `numpy` and `pandas` are used for numerical operations and data handling.
- `scikit-learn` models are used for anomaly detection.
- Evaluation metrics are imported to measure model performance.

This setup ensures the environment is ready for training and evaluation.

---

## Step 2: Label Preparation
Before training, labels are converted into a **binary format**:

- `0` → Normal data  
- `1` → Anomalous data (Attack / Fraud)

This conversion is required because machine learning evaluation metrics expect labels to be in numeric form.

---

## Step 3: Evaluation Function
A reusable evaluation function is created to compute:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (when anomaly scores are available)

This function ensures **consistent evaluation** across all models.

---

## Step 4: Isolation Forest Model Training
Isolation Forest is trained as the first baseline model.

- The model learns normal data patterns.
- Predictions are converted into binary anomaly labels.
- Anomaly scores are extracted for ROC-AUC calculation.

This model provides a **tree-based unsupervised baseline**.

---

## Step 5: Isolation Forest Evaluation
The Isolation Forest model is evaluated using:

- Classification metrics
- Confusion matrix

This establishes the **initial baseline performance**.

---

## Step 6: One-Class SVM Model Training
One-Class SVM is trained as the second baseline model.

- It learns a boundary around normal data points.
- Data points outside the boundary are classified as anomalies.

This model is sensitive to subtle anomalies and complements Isolation Forest.

---

## Step 7: One-Class SVM Evaluation
The One-Class SVM model is evaluated using the same metrics as Isolation Forest.

Using the same evaluation method allows fair comparison between models.

---

## Step 8: Autoencoder Model Definition
A neural network autoencoder is defined with:

- An encoder to compress data
- A decoder to reconstruct data

The model is designed to learn normal data reconstruction patterns.

---

## Step 9: Autoencoder Training
The autoencoder is trained using normal data only.

- The model learns to reconstruct normal samples accurately.
- No labels are used during training.

---

## Step 10: Autoencoder-Based Anomaly Detection
Anomalies are detected using reconstruction error:

- High reconstruction error indicates an anomaly.
- A threshold (95th percentile) is used to classify anomalies.

This provides a **reconstruction-based baseline model**.

---

## Step 11: Autoencoder Evaluation
The autoencoder predictions are evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

This allows comparison with other baseline models.

---

## Step 12: Storing Baseline Results
All model results are stored together for:

- Easy comparison
- Future ensemble construction
- Statistical analysis

This concludes the baseline model training phase.

---

## Summary
This notebook:
- Trains three baseline anomaly detection models
- Evaluates each model using consistent metrics
- Establishes reference performance for future work

The results from this notebook are used as **baseline benchmarks** for ensemble and GenAI-augmented models in later stages.
