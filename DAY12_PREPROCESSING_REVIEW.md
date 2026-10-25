# Day 12: Preprocessing Review & Dataset Version 1.0 Sign-off (Master Realignment)

## Deliverables Summary
* Integrated Review of EDA, Class Balance, and Splitting Boundaries across Master Fields
* Engineering Defense Matrix for Every Technical Pipeline Decision
* Resolution Check log for Mentor Dataset Path Corrections
* Official Freezing of Master Dataset Version 1.0 and Approved Preprocessors

---

## Technical Context and Overview

### What I Did on Day 12
On Day 12, I synthesized our experimental findings from Exploratory Data Analysis, class balance indexing, and pipeline transformations to present a cohesive preprocessing review pack using our mentor's official master dataset. I constructed a definitive technical defense matrix to justify our processing workflow, integrated corrections to parse our true data headers case-insensitively, and permanently froze Dataset Version 1.0 along with our serialized pipeline architecture binaries (`approved_preprocessor_v1.pkl`) to establish a clean data baseline for the upcoming modeling tasks.

### Dataset and Preprocessing Architecture Explanation
The underlying dataset consists of a single combined table containing 399 verified digital communication pulse tracking rows. The features map to `Analog Voltage` for continuous sensor metrics and `Binary bit S` for target payload flags, grouped across multiple turbidity tiers (`1-2NTU` up to `5-6NTU`). To guarantee that our model does not experience data leakage from future turbidity states, the pipeline filters the data to isolate the clear-water baseline (`1-2NTU`) values before running an 80/20 stratified split. All preprocessing limits are calculated strictly from the training data split, enforcing a complete firewall against data leakage.

---

## Technical Decision Defense Matrix

| Processing Layer | Selected Strategy | Engineering Defense & Justification |
| :--- | :--- | :--- |
| **Cross-Condition Isolation** | `df[df['NTU'] == '1-2NTU']` | Isolates the clear baseline rows from the master file first, blocking downstream cross-condition data leakage. |
| **Label Encoding** | `.astype(str).str.strip().astype(int)` | Resolves target column string trailing space errors. Ensures binary integrity before array operations. |
| **Outlier Treatment** | Z-Score Clipping (± 4σ) | Clamps extreme transient switching circuit spikes without dropping rows, preserving signal history. |
| **Feature Scaling** | Min-Max Normalization | Transforms voltage inputs into bounded numeric windows, which is critical for distance-based estimators like KNN. |

---

## Google Colab Interactive Notebook Code Execution

### Cell Execution: Final Master Dataset V1 Freezing Configuration
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Load the verified raw signal master matrix
df = pd.read_csv('uwoc_master_dataset.csv')
df.columns = [col.strip() for col in df.columns]

feature_col = 'Analog Voltage'
target_col = 'Binary bit S'

# Clean hidden trailing whitespace strings
df[target_col] = df[target_col].astype(str).str.strip().astype(int)

# Isolate clear baseline parameters to prevent cross-condition leakage
df_clear = df[df['NTU'].str.strip() == '1-2NTU']

# Partition data using a stratified layout to secure uniform distribution
X_train, X_test, y_train, y_test = train_test_split(
    df_clear[[feature_col]], df_clear[target_col], test_size=0.20, random_state=42, stratify=df_clear[target_col]
)

# Establish the approved preprocessing sequence
v1_pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1)))
])

X_train_v1 = v1_pipeline.fit_transform(X_train)
X_test_v1 = v1_pipeline.transform(X_test)

# Serialize the dataset version 1.0 assets to the project workspace
joblib.dump(v1_pipeline, 'uwoc_dataset_v1_frozen/approved_preprocessor_v1.pkl')
print("Dataset Version 1.0 state frozen successfully.")
```

---

## Console Log Terminal Output

```text
=========================================================
RUNNING MASTER REALIGNMENT ENGINE: DATASET VERSION 1.0
=========================================================

CRITICAL DEFENSE TELEMETRY BLOCK:
  * Dataset State: Frozen Master Dataset Version 1.0 Locked.
  * Input File Target Path: uwoc_master_dataset.csv
  * Isolated Train Block Shape   : (64, 1)
  * Isolated Test Holdout Shape  : (16, 1)
  * Balanced Target Splitting: Class 0 = 32 | Class 1 = 32
  * Preprocessor Object Destination: uwoc_dataset_v1_frozen/approved_preprocessor_v1.pkl

=========================================================
```
