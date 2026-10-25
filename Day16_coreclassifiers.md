# Day 16: Core Classifiers

## Deliverables Summary
* Implementation of Support Vector Machine (SVM) and Random Forest Core Classifiers
* Strict Execution Using Identical Dataset Version 1.0 Partitions and Evaluation Metrics
* Comprehensive Tracking Log of Core Model Hyperparameters and Output Metrics
* Direct Comparative Analysis of Class-Wise Model Performance and Confusion Matrices

---

## Technical Context and Overview

### What I Did on Day 16
On Day 16, I expanded our machine learning evaluation matrix by implementing and training two core classifiers: a non-linear Support Vector Machine (SVM) utilizing a Radial Basis Function (RBF) kernel and a Random Forest Ensemble restricted to a max depth of 3. To ensure mathematical consistency, both models were trained and scored using the exact same stratified data splits frozen on Day 12 from the master dataset array pool.

---

## Core Classifier Performance Comparison Matrix

| Algorithm Evaluated | Selected Hyperparameters | Accuracy | Precision | Recall | F1-Score | Bit Error Rate (BER) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine** | kernel='rbf', C=1.0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000e+00 |
| **Random Forest** | n_estimators=50, max_depth=3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000e+00 |

---

## Console Log Terminal Output

```text
=========================================================
RUNNING DAY 16: CORE CLASSIFIERS (SVM & RANDOM FOREST)
=========================================================

 CORE CLASSIFIERS METRICS SCOREBOARD:
---------------------------------------------------------
1. SUPPORT VECTOR MACHINE (SVM) EXPERIMENTAL LOG:
  * Accuracy         : 1.0000
  * Bit Error Rate   : 0.0000e+00
  * Confusion Matrix :
[[8 0]
 [0 8]]

2. RANDOM FOREST ENSEMBLE EXPERIMENTAL LOG:
  * Accuracy         : 1.0000
  * Bit Error Rate   : 0.0000e+00
  * Confusion Matrix :
[[8 0]
 [0 8]]
=========================================================
```
