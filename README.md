# Base Model Training – Anomaly Detection

## Overview
This module implements baseline anomaly detection models as described in the Aman research proposal. These models establish reference performance against which ensemble and GenAI-augmented methods are evaluated.

## Datasets
- NSL-KDD (Network Intrusion Detection)
- Credit Card Fraud Detection Dataset

Each dataset is processed and modeled separately to maintain domain validity.

## Models Implemented
1. Isolation Forest  
2. One-Class Support Vector Machine (OCSVM)  
3. Autoencoder (Neural Network)

All models are trained in an unsupervised manner.

## Label Convention
- 0 → Normal
- 1 → Anomaly (Attack / Fraud)

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Accuracy is reported but not relied upon due to class imbalance.

## Workflow
1. Preprocess dataset (encoding, normalization, splitting)
2. Train base anomaly detection models
3. Convert model outputs to binary labels
4. Evaluate using multiple metrics
5. Store results for ensemble comparison

## Purpose of Base Models
- Establish baseline anomaly detection performance
- Highlight limitations of single learners
- Justify the need for ensemble learning and GenAI integration

## Expected Behavior
- Moderate recall
- Low precision
- Reasonable ROC-AUC
These characteristics are typical for unsupervised anomaly detection models.

## Next Steps
- Build traditional ensemble models
- Integrate GenAI-based augmentation
- Perform statistical validation (ANOVA)

