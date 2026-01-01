```
1. Importing Libraries
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
```
Explanation

numpy, pandas: Used for numerical operations and data handling.

IsolationForest: Tree-based unsupervised anomaly detection model.

OneClassSVM: Boundary-based unsupervised anomaly detector.

sklearn.metrics: Used to evaluate model performance using standard classification metrics.

## Why needed?
Anomaly detection results must be evaluated quantitatively to compare base models with ensembles later.
```
2. Evaluation Function
def evaluate_model(y_true, y_pred, y_score=None):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)

    return metrics
```
Explanation

Converts labels to integers to avoid type mismatch errors.

Computes:

Accuracy → overall correctness

Precision → correctness of anomaly predictions

Recall → ability to detect actual anomalies

F1-score → balance between precision and recall

roc_auc is calculated only if anomaly scores are available.

## Why important?
Anomaly detection datasets are highly imbalanced, so multiple metrics are required.
```
3. Isolation Forest – Training
iso_forest = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)

iso_forest.fit(X_train)
```
Explanation

n_estimators=200: Number of trees → more trees = stable anomaly isolation.

contamination="auto": Automatically estimates anomaly proportion.

fit(X_train): Model learns normal data patterns.

## Why Isolation Forest?
It isolates anomalies using random splits and works well for high-dimensional data.
```
4. Isolation Forest – Prediction & Label Conversion
y_pred_if = iso_forest.predict(X_test)
y_pred_if = np.where(y_pred_if == -1, 1, 0)
```
Explanation

Isolation Forest outputs:

-1 → anomaly

+1 → normal

Converted to:

1 → anomaly

0 → normal

## Why convert?
All models must follow one unified label system for evaluation and ensembles.
```
5. Isolation Forest – Anomaly Scores
y_score_if = -iso_forest.decision_function(X_test)
```
Explanation

decision_function() gives distance from normality.

Negative sign ensures higher score = higher anomaly likelihood.

## Used for: ROC-AUC calculation and ensemble weighting.
```
6. Isolation Forest – Evaluation
if_metrics = evaluate_model(y_test, y_pred_if, y_score_if)

print(if_metrics)
print(confusion_matrix(y_test, y_pred_if))
```
Explanation

Computes all evaluation metrics.

Confusion matrix shows:

True Positives

False Positives

False Negatives

True Negatives

## Purpose: Establish baseline performance.
```
7. One-Class SVM – Training
ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

ocsvm.fit(X_train)
```
Explanation

rbf kernel: Captures non-linear boundaries.

nu=0.05: Expected anomaly proportion.

Learns boundary enclosing normal data.

## Why OCSVM?
It is sensitive to subtle anomalies but prone to false positives.
```
8. One-Class SVM – Prediction & Evaluation
y_pred_svm = ocsvm.predict(X_test)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)

y_score_svm = -ocsvm.decision_function(X_test)

svm_metrics = evaluate_model(y_test, y_pred_svm, y_score_svm)
```
Explanation

Same label conversion as Isolation Forest.

Decision scores used for ROC-AUC.

## Purpose: Compare boundary-based detection vs tree-based detection.
```
9. Autoencoder – Model Definition
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)
```
Explanation

Encoder compresses data.

Decoder reconstructs input.

Anomalies produce high reconstruction error.

##  Why Autoencoder?
Effective for complex, high-dimensional anomaly patterns.
```
 10. Autoencoder – Training
autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=256,
    validation_split=0.1,
    shuffle=True
)
```
Explanation

Trained only on normal data.

Learns reconstruction of normal behavior.
```
 11. Autoencoder – Anomaly Detection
reconstruction_error = np.mean(np.square(X_test - reconstructions), axis=1)
threshold = np.percentile(reconstruction_error, 95)
y_pred_ae = (reconstruction_error > threshold).astype(int)
```

Explanation

High reconstruction error → anomaly.

Threshold set at 95th percentile.

## Purpose: Reconstruction-based anomaly detection baseline.