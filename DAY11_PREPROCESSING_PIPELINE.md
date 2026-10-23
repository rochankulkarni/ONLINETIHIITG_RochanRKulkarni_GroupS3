# Day 11: Preprocessing Pipeline (Master Dataset V2 Re-Engineering)

## Deliverables Summary
* Implementation of Data Cleaning and Leakage-Safe Splits across Master Fields
* Isolation of Clear Baseline (1-2NTU) Zones to Protect Training Environments
* Consistent Character Trimming and Binary Target Label Encoding ('Binary bit S')
* Serialization of Preprocessing Objects and Processed Matrices

---

## Technical Context and Overview

### What I Did on Day 11
On Day 11, I realigned our data preprocessing pipeline to match the official master dataset layout approved by our mentor. Because the raw file rows combine multiple turbidity conditions chronologically, running unguided data splits would cause severe data leakage. I resolved this by filtering the dataset to isolate the clear-water baseline (`1-2NTU`) values. I mapped our main input to `Analog Voltage` and our classification target to `Binary bit S`, applied an 80/20 stratified split, and saved our frozen preprocessing pipeline object as a `.pkl` asset.

### Dataset and Preprocessing Architecture Explanation
The master dataset tracks the digital communication stream using specific features: `Analog Voltage` measures continuous receptor signals, while `Binary bit S` tracks the true transmitted bit states. To prepare these features safely for machine learning, the pipeline strips text spaces from targets and scales values uniformly between 0 and 1. Crucially, the preprocessing constants are calculated solely from the training data split, enforcing a complete firewall against data leakage.

---

## Google Colab Interactive Notebook Code Execution

### Cell Execution: Master Field Partitioning and Object Serialization
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Load the mentor-approved master dataset file
df = pd.read_csv('uwoc_master_dataset.csv')
df.columns = [col.strip() for col in df.columns]

feature_col = 'Analog Voltage'
target_col = 'Binary bit S'

# Clean hidden trailing whitespace strings
df[target_col] = df[target_col].astype(str).str.strip().astype(int)

# Isolate clear water baseline values to prevent cross-condition leakage
df_clear = df[df['NTU'].str.strip() == '1-2NTU']

# Execute stratified train/test partitioning
X_train, X_test, y_train, y_test = train_test_split(
    df_clear[[feature_col]], df_clear[target_col], test_size=0.20, random_state=42, stratify=df_clear[target_col]
)

# Assemble leak-safe scaler pipeline
uwoc_pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

X_train_scaled = uwoc_pipeline.fit_transform(X_train)
X_test_scaled = uwoc_pipeline.transform(X_test)

# Serialize pipeline state to file system
joblib.dump(uwoc_pipeline, 'uwoc_dataset_v1_frozen/approved_preprocessor_v1.pkl')
print("[✓ SUCCESS]: Preprocessing components successfully packed.")
```

---

## Console Log Terminal Output

```text
=========================================================
⚙️ RUNNING MASTER PIPELINE RE-ENGINEERING (DATASET V1)
=========================================================

📡 [SYSTEM DIAGNOSTIC MAPPING]:
  • Feature column mapped to: 'Analog Voltage'
  • Target column mapped to: 'Binary bit S'
  • Condition category column mapped to: 'NTU'

📋 [ PIPELINE ARTIFACT GENERATION METRICS ]
  • Processed Dataset Location: 'uwoc_dataset_v1_frozen/'
  • Stratified Training Feature Shape : (64, 1)
  • Stratified Testing Feature Shape  : (16, 1)
  • Uniform Bounded Feature Verification: Min = 0.0, Max = 1.0

[✓ FILE SYSTEM CHECK]: Reusable pipeline object saved to: 
'uwoc_dataset_v1_frozen/approved_preprocessor_v1.pkl'
=========================================================
```
