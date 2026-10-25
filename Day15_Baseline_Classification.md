# Day 15: Baseline Classification and Mentor Review

## Deliverables Summary
* Implementation of Fixed-Threshold and Logistic Regression Performance Baselines
* Comprehensive Evaluation Matrix: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
* Calculated Bit Error Rate (BER) System Parameters for the UWOC Signal Space
* Structural Presentation of Baseline Trends and Confirmed Modelling Directions

---

## Technical Context and Overview

### What I Did on Day 15
On Day 15, I established our initial model performance baselines using our locked Master Dataset Version 1.0 split configuration files. I simulated a static hardware Fixed-Threshold detector at the scaled 0.5 midpoint as our control rule. I then trained a regularized linear Logistic Regression model to act as our primary machine learning baseline. Both tracking structures were scored across six engineering metrics on the test holdout slice to measure baseline channel decoding accuracy under clear-water conditions (1-2 NTU).

---

## Google Colab Interactive Notebook Code Execution

### Cell Execution: Training Baselines and Calculating Channel Metrics
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train = np.load('uwoc_dataset_v1_frozen/X_train_v1.npy')
X_test = np.load('uwoc_dataset_v1_frozen/X_test_v1.npy')
y_test = pd.read_csv('uwoc_dataset_v1_frozen/y_test_v1.csv').iloc[:, 0].values

# Simulate static hard threshold detector
y_pred_static = np.where(X_test >= 0.5, 1, 0)
print(f"Static Rule Accuracy: {accuracy_score(y_test, y_pred_static):.4f}")
print(f"Static Confusion Matrix:\n{confusion_matrix(y_test, y_pred_static)}")
```

---

## Console Log Terminal Output

```text
=========================================================
RUNNING DAY 15: FIXED-THRESHOLD & LOGISTIC REGRESSION 
=========================================================

 BASELINE PERFORMANCE METRICS SCOREBOARD:
---------------------------------------------------------
1. FIXED-THRESHOLD HARDWARE BASELINE:
  * Accuracy         : 1.0000
  * Precision        : 1.0000
  * Recall           : 1.0000
  * F1-Score         : 1.0000
  * Bit Error Rate   : 0.0000e+00
  * Confusion Matrix :
[[8 0]
 [0 8]]

2. LOGISTIC REGRESSION MACHINE LEARNING BASELINE:
  * Accuracy         : 1.0000
  * Precision        : 1.0000
  * Recall           : 1.0000
  * F1-Score         : 1.0000
  * Bit Error Rate   : 0.0000e+00
  * Confusion Matrix :
[[8 0]
 [0 8]]
=========================================================
```
