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
```
def evaluate(y_true, y_pred, y_score):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

```

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
```
iso_forest = IsolationForest(
    n_estimators=300,
    contamination=0.003,
    random_state=42,
    n_jobs=-1,
    max_samples=0.8
)
iso_forest.fit(X_train)

iso_scores = -iso_forest.decision_function(X_test)
iso_pred = (iso_forest.predict(X_test) == -1).astype(int)

iso_results = evaluate(y_test, iso_pred, iso_scores)

```

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
```
ocsvm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.0017
)

ocsvm.fit(X_train_normal)

svm_scores = -ocsvm.decision_function(X_test)
svm_pred = (ocsvm.predict(X_test) == -1).astype(int)

svm_results = evaluate(y_test, svm_pred, svm_scores)

```
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

```
input_dim = X_train.shape[1]

inputs = Input(shape=(input_dim,))
x = Dense(64, activation="relu")(inputs)
x = Dense(32, activation="relu")(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(
    X_train_normal,
    X_train_normal,
    epochs=100,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

reconstructions = autoencoder.predict(X_test)
reconstruction_error = np.mean(
    np.square(X_test - reconstructions),
    axis=1
)

```

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
