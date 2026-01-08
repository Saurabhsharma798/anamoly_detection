# Anomaly Detection – Base Model Training

The goal of this stage is to establish **baseline performance** using single anomaly detection models before moving to ensemble and GenAI-based methods.

---

## Datasets Used
The experiments are performed separately on the following datasets:

1. **NSL-KDD Dataset**
   - Used for network intrusion (attack) detection
   - Contains both normal and attack traffic records
   - Mixed numerical and categorical features

2. **Credit Card Fraud Dataset**
   - Used for fraud detection
   - Highly imbalanced dataset
   - All features are numerical (mostly PCA-transformed)

Each dataset is handled **independently** to maintain domain correctness.

---

## Label Convention
For consistency across all models:

- `0` → Normal data
- `1` → Anomalous data (Attack / Fraud)

All labels are converted into this binary format before evaluation.

---

## Models Implemented (Base Models)

The following **single learner anomaly detection models** are trained:

### 1. Isolation Forest
- Tree-based unsupervised anomaly detection model
- Detects anomalies by isolating rare data points
- Suitable for high-dimensional data

### 2. One-Class Support Vector Machine (OCSVM)
- Learns a boundary around normal data
- Detects anomalies that fall outside this boundary
- Sensitive to subtle anomaly patterns

### 3. Autoencoder (Neural Network)
- Learns to reconstruct normal data
- Anomalies are detected using high reconstruction error
- Useful for capturing non-linear patterns

These models serve as **baseline learners**.

---

## Training Approach
- Models are trained in an **unsupervised manner**
- Training is primarily done on normal data patterns
- No class balancing or data augmentation is applied at this stage

The purpose is **not to achieve perfect performance**, but to create a reference point for comparison.

---

## Evaluation Metrics
Each model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

 **Note:**  
Accuracy is reported but not relied upon heavily due to class imbalance. Recall and F1-score are more important for anomaly detection.

---

## Observations
- Base models show moderate recall and low precision
- This behavior is expected for unsupervised anomaly detection
- Results highlight the limitations of single learners
- These limitations justify the use of ensemble and GenAI-based methods later

---

## Purpose of This Stage
- Establish baseline anomaly detection performance
- Compare different single learners
- Provide a foundation for:
  - Traditional ensemble methods
  - GenAI-augmented ensemble models
  - Statistical significance testing

---

## Next Steps
The next phases of the project will include:

1. Traditional ensemble models (Voting, Bagging, Boosting)
2. GenAI-based data augmentation
3. GenAI as an auxiliary learner
4. GenAI-powered ensemble models
5. Statistical validation using ANOVA

---

## Reproducibility
All experiments are conducted using Python and standard machine learning libraries.  
The workflow is modular and can be easily extended for ensemble and GenAI integration.

---

## Conclusion
This repository currently contains the **baseline model training and evaluation** required for anomaly detection research. These results act as a benchmark for evaluating improvements achieved through ensemble learning and Generative AI techniques.
