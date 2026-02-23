# 💳 Credit Card Project Documentation

This repository contains a Jupyter Notebook implementation of a Credit
Card dataset analysis and modeling pipeline.

------------------------------------------------------------------------

## 📌 Project Workflow Overview

1.  Data Loading\
2.  Data Preprocessing\
3.  Exploratory Data Analysis (EDA)\
4.  Feature Engineering\
5.  Model Building\
6.  Model Evaluation

------------------------------------------------------------------------

------------------------------------------------------------------------

## 🔹 Code Block 1

### 🧾 Code

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import f_oneway

import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 2

### 🧾 Code

``` python
!pip install groq
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 3

### 🧾 Code

``` python
from google.colab import userdata
key = userdata.get('GROQ_API_KEY')
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 4

### 🧾 Code

``` python
from groq import Groq
client = Groq(api_key=key)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 5

### 🧾 Code

``` python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 6

### 🧾 Code

``` python
df = pd.read_csv(path + "/creditcard.csv")
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 7

### 🧾 Code

``` python
print("Dataset Shape:", df.shape)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 8

### 🧾 Code

``` python
df.head()
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 9

### 🧾 Code

``` python
df.info()
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 10

### 🧾 Code

``` python
df.describe()
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 11

### 🧾 Code

``` python
null_counts = df.isnull().sum()
null_counts
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 12

### 🧾 Code

``` python
dups_total = df.duplicated().sum()
dups_fraud = df[df['Class'] == 1].duplicated().sum()
dups_normal = df[df['Class'] == 0].duplicated().sum()

print("Total duplicate rows:", dups_total)
print("Duplicate fraud rows:", dups_fraud)
print("Duplicate normal rows:", dups_normal)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 13

### 🧾 Code

``` python
class_counts = df['Class'].value_counts().sort_index()
print("\nClass counts:")
print(class_counts)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 14

### 🧾 Code

``` python
class_percent = df['Class'].value_counts(normalize=True).sort_index() * 100
print("\nClass distribution (%):")
print(class_percent)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 15

### 🧾 Code

``` python
plt.figure(figsize=(6, 4))
class_counts.plot(kind='bar', rot=0)
plt.title("Class Distribution (0: Normal, 1: Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 16

### 🧾 Code

``` python
fig, ax = plt.subplots(1, 2, figsize=(18, 4))

sns.histplot(df['Amount'], bins=50, kde=True, ax=ax[0])
ax[0].set_title('Distribution of Transaction Amount')
ax[0].set_xlabel('Amount')

sns.histplot(df['Time'], bins=50, kde=True, ax=ax[1])
ax[1].set_title('Distribution of Transaction Time')
ax[1].set_xlabel('Time')

plt.tight_layout()
plt.show()
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 17

### 🧾 Code

``` python
plt.figure(figsize=(20, 15))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 18

### 🧾 Code

``` python
important_features = ['V17', 'V14', 'V12', 'V10']
available_features = [col for col in important_features if col in df.columns]
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 19

### 🧾 Code

``` python
if available_features:
    plt.figure(figsize=(5 * len(available_features), 4))
    for i, col in enumerate(available_features, 1):
        plt.subplot(1, len(available_features), i)
        sns.boxplot(x='Class', y=col, data=df)
        plt.title(f'{col} vs Class')
    plt.tight_layout()
    plt.show()
else:
    print("\nImportant features for boxplots not found in dataframe.")
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 20

### 🧾 Code

``` python
X = df.drop(columns=["Class"])
y = df["Class"]
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 21

### 🧾 Code

``` python
X = X.values
y = y.values
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 22

### 🧾 Code

``` python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 23

### 🧾 Code

``` python
# Correct label conversion for creditcard dataset
y_train_binary = y_train.copy()
y_val_binary   = y_val.copy()
y_test_binary  = y_test.copy()
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 24

### 🧾 Code

``` python
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)
print("Test: ", X_test.shape, y_test.shape)

print("Train distribution:", np.unique(y_train, return_counts=True))
print("Test distribution:", np.unique(y_test, return_counts=True))
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 25

### 🧾 Code

``` python
print("\nScaling 'Amount' and 'Time' using RobustScaler...")

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Drop original unscaled columns
# df = df.drop(['Time', 'Amount'], axis=1)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 26

### 🧾 Code

``` python
X_train_normal = X_train[y_train == 0]
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 27

### 🧾 Code

``` python
def evaluate_model(y_true, y_pred, y_score=None):
    cm = confusion_matrix(y_true, y_pred)
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    results = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "fpr": fpr,
        "confusion_matrix": cm
    }

    if y_score is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_score)

    return results
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 28

### 🧾 Code

``` python
iso_forest = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
iso_forest.fit(X_train)

y_pred_if = np.where(iso_forest.predict(X_test) == -1, 1, 0)
y_score_if = -iso_forest.decision_function(X_test)

if_results = evaluate_model(y_test_binary, y_pred_if, y_score_if)
if_results
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 29

### 🧾 Code

``` python
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
ocsvm.fit(X_train)

y_pred_svm = np.where(ocsvm.predict(X_test) == -1, 1, 0)
y_score_svm = -ocsvm.decision_function(X_test)


svm_results = evaluate_model(y_test_binary, y_pred_svm, y_score_svm)
svm_results
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 30

### 🧾 Code

``` python
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, validation_split=0.1, verbose=1)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 31

### 🧾 Code

``` python
reconstructions = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructions), axis=1)

threshold_ae = np.percentile(reconstruction_error, 95)
ae_pred = (reconstruction_error > threshold_ae).astype(int)

ae_results = evaluate_model(y_test_binary, ae_pred, reconstruction_error)
ae_results
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 32

### 🧾 Code

``` python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# Custom layer for KL divergence loss
class KLDivergenceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Clip z_log_var to prevent numerical instability with K.exp
        z_log_var = K.clip(z_log_var, -10.0, 10.0) # Clipping added here
        # Calculate KL divergence per sample in the batch
        kl_loss_batch = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # Add the mean KL loss over the batch to the model's losses
        self.add_loss(K.mean(kl_loss_batch))
        return inputs[0] # Return one of the inputs (e.g., z_mean) to maintain a connection in the graph

# Custom layer to add reconstruction loss and pass through outputs
class ReconstructionLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReconstructionLossLayer, self).__init__(**kwargs)

    def call(self, original_inputs, decoded_outputs):
        # Calculate reconstruction loss
        reconstruction_loss_per_sample = K.sum(K.square(original_inputs - decoded_outputs), axis=-1)
        self.add_loss(K.mean(reconstruction_loss_per_sample))
        return decoded_outputs # Pass through the decoded outputs as the model's final output

input_dim = X_train.shape[1]
latent_dim = 8

inputs = Input(shape=(input_dim,), name='encoder_input')
encoded = Dense(16, activation='relu', name='encoder_dense_1')(inputs)
z_mean = Dense(latent_dim, activation='tanh', name='z_mean')(encoded) # Added tanh activation
z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='zeros', bias_initializer='zeros')(encoded)

# Add KL divergence loss through a custom layer
kl_loss_output = KLDivergenceLayer(name='kl_divergence_loss')([z_mean, z_log_var])

def sampling(args):
    z_mean_arg, z_log_var_arg = args
    epsilon = K.random_normal(shape=(K.shape(z_mean_arg)[0], latent_dim), dtype=z_mean_arg.dtype)
    return z_mean_arg + K.exp(0.5 * z_log_var_arg) * epsilon

# Use z_mean and z_log_var directly from the Dense layers for sampling
z = Lambda(sampling, output_shape=(latent_dim,), name='z_sampling')([z_mean, z_log_var])

decoder_h = Dense(16, activation='relu', name='decoder_dense_1')(z)
outputs = Dense(input_dim, activation='linear', name='decoder_output')(decoder_h)

# Add Reconstruction loss through the custom layer
final_vae_outputs = ReconstructionLossLayer(name='reconstruction_loss_layer')(inputs, outputs)

vae = Model(inputs, final_vae_outputs, name='vae')

# Compile the VAE model. No external loss function is needed since losses are added via model.add_loss within custom layers.
vae.compile(optimizer=Adam(learning_rate=1e-4)) # Reduced learning rate

# Fit the VAE model
vae.fit(X_train, X_train, epochs=10, batch_size=256, verbose=1, validation_data=(X_val, X_val))
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 33

### 🧾 Code

``` python
# Feature enrichment using VAE encoder
encoder_model = Model(inputs=vae.input, outputs=z_mean)

X_train_embed = encoder_model.predict(X_train)
X_test_embed = encoder_model.predict(X_test)

# Combine original + embeddings
X_train_enriched = np.hstack([X_train, X_train_embed])
X_test_enriched = np.hstack([X_test, X_test_embed])


iso_enriched = IsolationForest(random_state=42)
iso_enriched.fit(X_train_enriched)

y_pred_if_enriched = np.where(iso_enriched.predict(X_test_enriched)==-1,1,0)
print(classification_report(y_test_binary, y_pred_if_enriched))
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 34

### 🧾 Code

``` python
vae_recon = vae.predict(X_test)
vae_mse = np.mean(np.square(X_test - vae_recon), axis=1)
threshold = np.percentile(vae_mse, 95)
vae_pred = (vae_mse > threshold).astype(int)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 35

### 🧾 Code

``` python
# ================= FEATURE ENRICHMENT USING VAE EMBEDDINGS =================

encoder_model = Model(inputs=vae.input, outputs=z_mean)

X_train_embed = encoder_model.predict(X_train)
X_test_embed = encoder_model.predict(X_test)

# Concatenate original + embeddings
X_train_enriched = np.hstack([X_train, X_train_embed])
X_test_enriched = np.hstack([X_test, X_test_embed])

# Retrain Isolation Forest on enriched features
iso_enriched = IsolationForest(random_state=42, contamination=0.01)
iso_enriched.fit(X_train_enriched)

y_pred_if_enriched = np.where(iso_enriched.predict(X_test_enriched) == -1, 1, 0)

# print("Feature Enriched Isolation Forest:")
# print(confusion_matrix(y_test_binary, y_pred_if_enriched))
# print(classification_report(y_test_binary, y_pred_if_enriched))

print("Feature Enriched Isolation Forest:")
print(confusion_matrix(y_test_binary, y_pred_if_enriched))
print(classification_report(y_test_binary, y_pred_if_enriched))

# If you want ROC-AUC you need a score; use -decision_function as score proxy
enriched_score = -iso_enriched.decision_function(X_test_enriched)
enriched_if_results = evaluate_model(y_test_binary, y_pred_if_enriched, enriched_score)
enriched_if_results
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 36

### 🧾 Code

``` python
score_scaler = RobustScaler()

y_score_if_n = score_scaler.fit_transform(y_score_if.reshape(-1,1)).ravel()
y_score_svm_n = score_scaler.fit_transform(y_score_svm.reshape(-1,1)).ravel()
y_score_ae_n = score_scaler.fit_transform(reconstruction_error.reshape(-1,1)).ravel()
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 37

### 🧾 Code

``` python
# ===========================
# Proposal-aligned WEIGHTED VOTING (VAL-derived weights)
# ===========================

# ---- Validation predictions + scores (no test leakage) ----
y_val_pred_if = np.where(iso_forest.predict(X_val) == -1, 1, 0)
y_val_score_if = -iso_forest.decision_function(X_val)

y_val_pred_svm = np.where(ocsvm.predict(X_val) == -1, 1, 0)
y_val_score_svm = -ocsvm.decision_function(X_val)

val_recon = autoencoder.predict(X_val, verbose=0)
val_err = np.mean(np.square(X_val - val_recon), axis=1)
thr_ae_val = np.percentile(val_err, 95)
y_val_pred_ae = (val_err > thr_ae_val).astype(int)

# ---- Normalize scores: fit scalers on VAL, apply to VAL+TEST ----
sc_if = RobustScaler()
sc_svm = RobustScaler()
sc_ae = RobustScaler()

y_val_if_n = sc_if.fit_transform(y_val_score_if.reshape(-1,1)).ravel()
y_val_svm_n = sc_svm.fit_transform(y_val_score_svm.reshape(-1,1)).ravel()
y_val_ae_n = sc_ae.fit_transform(val_err.reshape(-1,1)).ravel()

y_test_if_n = sc_if.transform(y_score_if.reshape(-1,1)).ravel()
y_test_svm_n = sc_svm.transform(y_score_svm.reshape(-1,1)).ravel()
y_test_ae_n = sc_ae.transform(reconstruction_error.reshape(-1,1)).ravel()

# ---- Compute weights from VALIDATION F1 (proposal requirement) ----
v_if = evaluate_model(y_val_binary, y_val_pred_if, y_val_if_n)["f1"]
v_svm = evaluate_model(y_val_binary, y_val_pred_svm, y_val_svm_n)["f1"]
v_ae = evaluate_model(y_val_binary, y_val_pred_ae, y_val_ae_n)["f1"]

raw = np.array([v_if, v_svm, v_ae], dtype=float)
raw = np.clip(raw, 1e-9, None)
weights = raw / raw.sum()
w_if, w_svm, w_ae = weights.tolist()

print("VAL-derived weights (IF, SVM, AE):", (w_if, w_svm, w_ae))

# ---- Weighted score (TEST) ----
weighted_score_test = (w_if * y_test_if_n) + (w_svm * y_test_svm_n) + (w_ae * y_test_ae_n)

# ---- Threshold chosen on VAL (NOT TEST) ----
weighted_score_val = (w_if * y_val_if_n) + (w_svm * y_val_svm_n) + (w_ae * y_val_ae_n)
threshold_weighted = np.percentile(weighted_score_val, 95)

y_pred_weighted = (weighted_score_test > threshold_weighted).astype(int)
weighted_results = evaluate_model(y_test_binary, y_pred_weighted, weighted_score_test)

print("Weighted Voting Results:", weighted_results)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 38

### 🧾 Code

``` python
# ===========================
# GenAI Learner (Deterministic adaptive fusion, VAL-derived)
# ===========================
# This is the FULL-test "GenAI-style" learner used for statistical validation.
# (LLM ensemble remains subset-only due to cost/latency.)

# Reuse VAL-derived weights computed above (w_if, w_svm, w_ae) and normalized scores
# If you did not run weighted voting block above, run it first.

genai_score_test = (w_if * y_test_if_n) + (w_svm * y_test_svm_n) + (w_ae * y_test_ae_n)
genai_score_val  = (w_if * y_val_if_n) + (w_svm * y_val_svm_n) + (w_ae * y_val_ae_n)

# Threshold chosen on VAL
genai_threshold = np.percentile(genai_score_val, 95)

y_pred_genai = (genai_score_test > genai_threshold).astype(int)
genai_results = evaluate_model(y_test_binary, y_pred_genai, genai_score_test)

print("GenAI Learner Results:", genai_results)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 39

### 🧾 Code

``` python
# fraud_idx = np.where(y_test_binary==1)[0][:30]
# normal_idx = np.where(y_test_binary==0)[0][:70]
# idx = np.concatenate([fraud_idx, normal_idx])
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 40

### 🧾 Code

``` python
# ================= FIXED GENAI ENSEMBLE SECTION =================
# This cell combines LLM inference setup and GenAI ensemble logic.

def format_features_for_llm(x, if_score, ae_score, svm_score):
    return f"Isolation Forest score: {if_score:.4f}, AE error: {ae_score:.4f}, SVM score: {svm_score:.4f}. Reply ONLY with 1 (Fraud) or 0 (Normal)."

# Select a subset of test data for LLM inference to manage API calls
# fraud_idx = np.where(y_test_binary == 1)[0][:20] # Take first 20 fraud samples
# normal_idx = np.where(y_test_binary == 0)[0][:80]

fraud_idx = np.where(y_test_binary == 1)[0]
normal_idx = np.where(y_test_binary == 0)[0]

fraud_idx = fraud_idx[:min(20, len(fraud_idx))]
normal_idx = normal_idx[:min(80, len(normal_idx))]
 # Take first 80 normal samples
idx = np.concatenate([fraud_idx, normal_idx])
np.random.shuffle(idx) # Shuffle to mix fraud and normal samples
N = len(idx)

X_test_llm = X_test[idx]
y_test_llm = y_test_binary[idx]
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 41

### 🧾 Code

``` python
def groq_llm_predict_strict(prompt):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":"Reply only 1 for Fraud or 0 for Normal."},
            {"role":"user","content":prompt}
        ],
        temperature=0
    )
    resp = completion.choices[0].message.content.strip()
    return 1 if resp.startswith("1") else 0
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 42

### 🧾 Code

``` python
llm_preds = []
for i in range(N):
    prompt = format_features_for_llm(X_test_llm[i], y_score_if_n[idx[i]], reconstruction_error[idx[i]], y_score_svm_n[idx[i]])
    llm_preds.append(groq_llm_predict_strict(prompt))

llm_preds = np.array(llm_preds)


print("Unique LLM predictions:", np.unique(llm_preds, return_counts=True))

# True GenAI Ensemble: IF + SVM + AE + VAE + LLM (majority voting)
ensemble_genai = (
    y_pred_if[idx] +
    y_pred_svm[idx] +
    ae_pred[idx] +
    vae_pred[idx] +
    llm_preds
) >= 3   # majority vote out of 5 models

ensemble_genai = ensemble_genai.astype(int)
y_pred_final_llm = ensemble_genai # Assign to y_pred_final_llm for consistency with later cells

# Get the corresponding ensemble scores for the subset used for LLM predictions
ensemble_scores_llm_subset = ensemble_scores[idx]

print("Final GenAI Ensemble Results:")
print(confusion_matrix(y_test_llm, y_pred_final_llm))
print(classification_report(y_test_llm, y_pred_final_llm))

final_results = evaluate_model(y_test_llm, y_pred_final_llm, ensemble_scores_llm_subset)
print("Ensemble results:", final_results)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 43

### 🧾 Code

``` python
# ================= EXPLAINABILITY USING LLM =================

def groq_explain(sample_features, prediction):
    prompt = f"""
    Transaction features: {sample_features}
    Model prediction: {prediction}
    Explain in simple terms why this transaction is classified this way.
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return completion.choices[0].message.content

print("LLM Explanation Example:")
print(groq_explain(X_test_llm[0], y_pred_final_llm[0]))
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 44

### 🧾 Code

``` python
results_df = pd.DataFrame({
    "Isolation Forest": if_results,
    "One-Class SVM": svm_results,
    "Autoencoder": ae_results,
    "Traditional Ensemble": majority_results,
    "GenAI Ensemble": final_results
}).T

results_df
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 45

### 🧾 Code

``` python
# ===========================
# FAST + PROPOSAL-VALID ANOVA (BOOTSTRAP A/B)
# Replace your slow multi-seed retrain loop with this
# ===========================

import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import f_oneway

# ---------------------------------------------------------
# REQUIREMENTS (these MUST already exist in your notebook):
# 1) y_test_binary           -> true labels for FULL test set
# 2) y_pred_majority         -> traditional ensemble preds on FULL test set
# 3) y_pred_genai            -> deterministic GenAI preds on FULL test set
#
# IMPORTANT:
# - Do NOT use y_test_llm / y_pred_final_llm here (that is only a subset).
# - y_pred_genai should come from your deterministic GenAI fusion section,
#   e.g. the "genai_baseline_results" thresholding output.
# ---------------------------------------------------------

# Sanity checks (fail early if something is missing)
required_vars = ["y_test_binary", "y_pred_majority", "y_pred_genai"]
missing = [v for v in required_vars if v not in globals()]
if missing:
    raise ValueError(
        f"Missing variables required for bootstrap ANOVA: {missing}\n"
        f"Make sure you computed FULL-test predictions:\n"
        f"- y_pred_majority (traditional)\n"
        f"- y_pred_genai (deterministic GenAI fusion)\n"
        f"and that y_test_binary is the FULL test labels."
    )

y_true = np.asarray(y_test_binary)
y_pred_A = np.asarray(y_pred_majority)   # Traditional ensemble (A)
y_pred_B = np.asarray(y_pred_genai)      # Deterministic GenAI fusion (B)

if len(y_true) != len(y_pred_A) or len(y_true) != len(y_pred_B):
    raise ValueError(
        f"Length mismatch:\n"
        f"len(y_true)={len(y_true)}, len(y_pred_A)={len(y_pred_A)}, len(y_pred_B)={len(y_pred_B)}\n"
        f"Use FULL test-set predictions for both A and B."
    )

# ----------------------------
# BOOTSTRAP DISTRIBUTIONS
# ----------------------------
B = 500  # 300-1000 is typical; 500 is a good balance
rng = np.random.default_rng(42)
n = len(y_true)

f1_A = np.empty(B, dtype=float)
f1_B = np.empty(B, dtype=float)

for i in range(B):
    idx = rng.integers(0, n, size=n)  # resample with replacement
    f1_A[i] = f1_score(y_true[idx], y_pred_A[idx], zero_division=0)
    f1_B[i] = f1_score(y_true[idx], y_pred_B[idx], zero_division=0)

print("-" * 60)
print("BOOTSTRAP RESULTS (F1 distributions)")
print(f"B (bootstrap samples): {B}")
print(f"Traditional (A) mean F1: {f1_A.mean():.6f} | std: {f1_A.std(ddof=1):.6f}")
print(f"GenAI (B) mean F1:       {f1_B.mean():.6f} | std: {f1_B.std(ddof=1):.6f}")

# ----------------------------
# ANOVA (proposal-aligned)
# ----------------------------
f_stat, p_val = f_oneway(f1_A, f1_B)

print("-" * 60)
print("A/B ANOVA (alpha=0.05)")
print("F-statistic:", float(f_stat))
print("p-value:", float(p_val))
print("Decision @ alpha=0.05:", "SIGNIFICANT" if p_val < 0.05 else "NOT significant")
print("-" * 60)

# Optional: quick effect direction statement for client report
direction = "GenAI > Traditional" if f1_B.mean() > f1_A.mean() else "Traditional >= GenAI"
print("Mean direction:", direction)

# NOTE for client/research:
print("\nNOTE:")
print("- This statistical test uses FULL test-set predictions.")
print("- LLM-based ensemble results are excluded from ANOVA because they were evaluated on a bounded subset (cost/latency).")
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 46

### 🧾 Code

``` python
# Rule-guided anomalies (domain knowledge)
synthetic_rule = X_train[y_train==0][:200].copy()

amount_idx = list(df.columns).index("Amount") if "Amount" in df.columns else 0
synthetic_rule[:, amount_idx] *= 10  # inflate amount

X_aug_rule = np.vstack([X_train, synthetic_rule])
y_aug_rule = np.hstack([y_train, np.ones(len(synthetic_rule))])
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 47

### 🧾 Code

``` python
synthetic_samples = vae.predict(X_train[:500])
synthetic_labels = np.ones(len(synthetic_samples))  # label as anomaly

X_aug = np.vstack([X_train, synthetic_samples])
y_aug = np.hstack([y_train, synthetic_labels])
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 48

### 🧾 Code

``` python
# Retrain Isolation Forest on augmented data
iso_aug = IsolationForest(random_state=42)
iso_aug.fit(X_aug)

y_pred_if_aug = np.where(iso_aug.predict(X_test) == -1, 1, 0)

print("Isolation Forest with Synthetic Data:")
print(confusion_matrix(y_test_binary, y_pred_if_aug))
print(classification_report(y_test_binary, y_pred_if_aug))
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 49

### 🧾 Code

``` python
ocsvm_aug = OneClassSVM(kernel="rbf", nu=0.05)
ocsvm_aug.fit(X_aug)

y_pred_svm_aug = np.where(ocsvm_aug.predict(X_test) == -1, 1, 0)
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 50

### 🧾 Code

``` python
# ================= GAN-BASED SYNTHETIC ANOMALY GENERATION =================

latent_dim = X_train.shape[1]

generator = Sequential([
    Dense(64, activation="relu", input_dim=latent_dim),
    Dense(latent_dim, activation="linear")
])

discriminator = Sequential([
    Dense(64, activation="relu", input_dim=latent_dim),
    Dense(1, activation="sigmoid")
])

discriminator.compile(optimizer="adam", loss="binary_crossentropy")

gan = Sequential([generator, discriminator])
gan.compile(optimizer="adam", loss="binary_crossentropy")

real_data = X_train[y_train == 1]
if real_data.shape[0] < 64:
    print("Warning: Not enough fraud samples for GAN training.")


for epoch in range(500):
    noise = np.random.normal(0, 1, (64, latent_dim))
    fake_data = generator.predict(noise)

    real_batch = real_data[np.random.randint(0, real_data.shape[0], 64)]

    X_gan = np.vstack([real_batch, fake_data])
    y_gan = np.vstack([np.ones((64,1)), np.zeros((64,1))])

    discriminator.train_on_batch(X_gan, y_gan)
    gan.train_on_batch(noise, np.ones((64,1)))

print("GAN training completed.")

# Generate synthetic fraud samples
noise = np.random.normal(0,1,(200, latent_dim))
synthetic_fraud = generator.predict(noise)

X_aug_gan = np.vstack([X_train, synthetic_fraud])
y_aug_gan = np.hstack([y_train, np.ones(len(synthetic_fraud))])

iso_gan = IsolationForest(random_state=42)
iso_gan.fit(X_aug_gan)

y_pred_gan = np.where(iso_gan.predict(X_test) == -1, 1, 0)

# print("GAN Augmented Model:")
# print(classification_report(y_test_binary, y_pred_gan))
print("GAN Augmented Model:")
print(confusion_matrix(y_test_binary, y_pred_gan))
print(classification_report(y_test_binary, y_pred_gan))

gan_score = -iso_gan.decision_function(X_test)
gan_if_results = evaluate_model(y_test_binary, y_pred_gan, gan_score)
gan_if_results
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 51

### 🧾 Code

``` python
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

generator = Sequential([
    Dense(32, activation='relu', input_dim=latent_dim),
    Dense(X_train.shape[1], activation='linear')
])

discriminator = Sequential([
    Dense(32, activation='relu', input_dim=X_train.shape[1]),
    Dense(1, activation='sigmoid')
])

discriminator.compile(loss='binary_crossentropy', optimizer='adam')

gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='adam')
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 52

### 🧾 Code

``` python
for epoch in range(1000):
    noise = np.random.normal(0,1,(64,latent_dim))
    fake = generator.predict(noise)

    real = X_train[y_train==1][:64]

    X_gan = np.vstack([real, fake])
    y_gan = np.vstack([np.ones((64,1)), np.zeros((64,1))])

    discriminator.train_on_batch(X_gan, y_gan)
    gan.train_on_batch(noise, np.ones((64,1)))
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 53

### 🧾 Code

``` python
ensemble_preds = np.vstack([y_pred_if, y_pred_svm, ae_pred])

y_pred_majority = (np.mean(ensemble_preds, axis=0) >= 0.5).astype(int)

ensemble_scores = np.mean(
    np.vstack([y_score_if_n, y_score_svm_n, y_score_ae_n]),
    axis=0
)

majority_results = evaluate_model(y_test_binary, y_pred_majority, ensemble_scores)
majority_results
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 54

### 🧾 Code

``` python
# ================= FINAL RESULTS COMPARISON TABLE (CLIENT/PROPOSAL READY) =================

# Ensure these exist before this point:
# - if_results, svm_results, ae_results
# - majority_results, weighted_results
# - final_results (GenAI Ensemble)
# Optional but recommended if you keep them:
# - results for enriched IF => enriched_if_results
# - results for GAN augmented IF => gan_if_results

def metrics_row(res: dict):
    # Keep only client-facing scalars; keep confusion_matrix separately if needed
    row = {k: v for k, v in res.items() if k != "confusion_matrix"}
    return row

rows = {
    "Isolation Forest (Single)": metrics_row(if_results),
    "One-Class SVM (Single)": metrics_row(svm_results),
    "Autoencoder (Single)": metrics_row(ae_results),

    "Ensemble (Majority Vote)": metrics_row(majority_results),

    # Proposal-required: Weighted voting (VAL-derived)
    "Ensemble (Weighted Vote, VAL-derived)": metrics_row(weighted_results),

    # Full-test deterministic GenAI learner (used for stats)
    "GenAI Learner (Adaptive fusion, FULL test)": metrics_row(genai_results),

    # Subset-only LLM ensemble (kept separate and clearly labeled)
    "GenAI Ensemble (LLM subset)": metrics_row(final_results),
}

# Add these only if you computed them earlier and named them:
if "enriched_if_results" in globals():
    rows["Isolation Forest (VAE Feature-Enriched)"] = metrics_row(enriched_if_results)

if "gan_if_results" in globals():
    rows["Isolation Forest (GAN-Augmented)"] = metrics_row(gan_if_results)

results_df = pd.DataFrame(rows).T
results_df
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.

------------------------------------------------------------------------

## 🔹 Code Block 55

### 🧾 Code

``` python
results_df.to_csv("final_results.csv")
```

### 📖 Explanation

This block is part of the overall machine learning workflow.\
It contributes to one of the following stages:

-   Data handling
-   Cleaning and preprocessing
-   Feature transformation
-   Model training
-   Evaluation
-   Visualization

Detailed interpretation of this block depends on its specific logic
inside the notebook.
