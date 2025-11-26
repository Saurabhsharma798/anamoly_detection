This project performs data loading, exploratory data analysis (EDA), preprocessing, feature engineering, and train–test splitting using the NSL-KDD intrusion detection dataset.



```
import numpy as np 
import pandas as pd
```
Loads essential libraries for numerical operations and data handling.



```
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

Lists all files available inside the /kaggle/input directory.


```
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
```

Imports all required Python libraries for preprocessing, visualization, scaling, and splitting.




```
columns = [...]
```

Defines the correct 43 column names for NSL-KDD dataset (41 features + attack_name + label).




```
df_train = pd.read_csv("/kaggle/input/nslkdd/KDDTrain+.txt", names=columns, sep=",", engine="python")
df_test = pd.read_csv("/kaggle/input/nslkdd/KDDTest+.txt", names=columns, sep=",", engine="python")

df = pd.concat([df_train, df_test], ignore_index=True)
```

Loads the training and testing files, assigns column names, and merges them into one dataframe.




```
df.head()
```

Displays the first few rows of the dataset.




```
df.info()
```

Shows data types, column counts, and memory usage.




```
df.describe()
```

Computes summary statistics for all numeric columns.




```
df.isnull().sum()
```

Checks missing values in each column.




```
categorical_cols = ["protocol_type", "service", "flag", "attack_name"]

numeric_cols = [c for c in df.columns if c not in categorical_cols + ["label"]]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```

Identifies categorical and numeric columns, then converts numeric columns to proper numeric types.




```
df["attack_name"].value_counts()
```

Shows the number of samples for each attack type.




```
print(df["flag"].unique()[:10])
```

Displays the first ten unique values of the flag column.




```
df["flag"] = df["flag"].astype(str)
categorical_cols = ["protocol_type", "service", "flag"]
```

Ensures flag is stored as string and updates the categorical column list.




```
df["binary_label"] = df["attack_name"].apply(
    lambda x: "normal" if x == "normal" else "attack"
)
```

Creates a binary label column separating normal vs attack traffic.




```
df["binary_label"].value_counts()
```

Counts occurrences of normal and attack samples.




```
df["binary_label"].value_counts().plot(kind="bar")
plt.title("Normal vs Attack Distribution")
plt.show()
```

Plots distribution of normal vs attack classes.




```
df["category"] = df["attack_name"].apply(classify)
```

Maps attack types into 5 categories (DoS, Probe, R2L, U2R, Other, Normal).




```
df["category"].value_counts().plot(kind="bar", color="violet")
plt.title("Attack Category Distribution")
plt.show()
```

Plots counts of each attack category.




```
df["category"].value_counts()
df["protocol_type"].value_counts()
df["service"].value_counts().head(10)
df["flag"].value_counts()
```

Displays value counts for main categorical features.




```
plt.figure(figsize=(6,4))
sns.countplot(x=df["protocol_type"])
plt.title("Protocol Type Distribution")
plt.show()
```

Plots distribution of protocol types (tcp/udp/icmp).





```
df["service"].value_counts().head(10).plot(kind="bar", figsize=(8,4))
plt.title("Top 10 Services")
plt.show()
```

Plots most frequent service types.





```
sns.countplot(y=df["flag"], order=df["flag"].value_counts().index)
plt.title("Flag Distribution")
plt.show()
```

Plots distribution of flag values.





```
plt.figure(figsize=(18,15))
numeric_only = df.select_dtypes(include=["int64","float64"])
sns.heatmap(numeric_only.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

Creates correlation heatmap of all numeric features.





```
for col in ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=50)
    plt.title(f"Distribution of {col}")
    plt.show()
```

Plots distribution of selected key numeric variables.





```
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

Applies one-hot encoding to protocol_type, service, and flag.





```
numerical_cols = [
    col for col in df.columns
    if col not in [
        "attack_name", "label", "binary_label", "category"
    ]
    and not col.startswith("protocol_type_")
    and not col.startswith("service_")
    and not col.startswith("flag_")
]
```

Recomputes numeric columns after one-hot encoding.




```
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

Scales all numeric columns using z-score normalization.





```
X = df.drop(["attack_name","label","binary_label","category"], axis=1)
y = df["binary_label"]
```

Defines feature set (X) and target variable (y).




```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=21
)
```

Splits data into stratified train/test sets.
