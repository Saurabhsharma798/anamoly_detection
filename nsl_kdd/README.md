
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

## Step 13: Create an Evaluation Function for Ensemble Models

In this step, a common evaluation function is defined to measure the performance of ensemble models.

The function calculates Accuracy, Precision, Recall, and F1-score.

It also calculates ROC-AUC when anomaly scores are provided.

This ensures that all ensemble models are evaluated using the same criteria.

Using a single evaluation function makes performance comparison fair and consistent.

---

## Step 14: Combine Predictions from All Models

Predicted labels from different anomaly detection models (Isolation Forest, One-Class SVM, and Autoencoder) are collected.

Predictions from each model are stacked together.

Each row represents one model’s predictions.

This step prepares the outputs for ensemble decision-making.

--

## Step 15: Apply Majority Voting to Get Final Predictions

Majority voting is used to decide the final class label for each sample.

If most models predict a sample as anomalous, it is labeled as an anomaly.

If most models predict it as normal, it is labeled as normal.

This approach reduces the effect of individual model errors.

---

## Step 16: Combine Anomaly Scores for ROC-AUC Calculation

Anomaly scores from all individual models are combined.

Scores are averaged to produce a single ensemble anomaly score.

This combined score is used to calculate ROC-AUC.

Using averaged scores provides a smoother and more reliable ranking of anomalies.

---

## Step 17: Evaluate Majority Voting Ensemble Performance

The majority voting ensemble predictions are evaluated.

Accuracy, Precision, Recall, and F1-score are calculated.

ROC-AUC is computed using the combined anomaly score.

This step shows whether the ensemble improves performance compared to individual models.

---

## Step 18: Define Weights for Weighted Ensemble Models

Different weights are assigned to each anomaly detection model.

Models with better performance are given higher weights.

Weights control how much each model influences the final decision.

This allows the ensemble to rely more on stronger models.

---

Step 19: Calculate Weighted Anomaly Score

A weighted anomaly score is computed using the assigned weights.

Each model’s anomaly score is multiplied by its weight.

All weighted scores are summed to obtain a final score.

This creates a more balanced and performance-aware anomaly score.

---

## Step 20: Select Threshold for Weighted Ensemble

A threshold is chosen using a percentile-based method.

Samples with scores above the threshold are classified as anomalies.

The threshold can be tuned to control false positives and false negatives.

This step converts anomaly scores into class labels.

---

## Step 21: Evaluate Weighted Voting Ensemble Performance

The weighted ensemble predictions are evaluated.

Classification metrics such as Accuracy, Precision, Recall, and F1-score are calculated.

ROC-AUC is computed using weighted anomaly scores.

This evaluation helps determine if weighting improves detection performance.

---

## Step 22: Compare Ensemble Model Results

Performance metrics from:

Majority Voting Ensemble

Weighted Voting Ensemble

are collected and displayed in a table.

This comparison helps identify the most effective ensemble strategy.



## Summary
This notebook:
- Trains three base anomaly detection models
- Evaluates each model using consistent metrics
- Establishes baseline performance for anomaly detection

The results from this notebook serve as a **reference point** for further improvements using ensemble and GenAI-based methods.
