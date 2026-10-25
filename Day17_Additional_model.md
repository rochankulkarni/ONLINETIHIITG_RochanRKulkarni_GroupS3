# Day 17: Additional Model Implementation

## Deliverables Summary
* Implementation and Training of a Small Multi-Layer Perceptron (MLP) Classifier
* Comprehensive Tracking of Neural Network Validation Behavior Across Training Eras
* Direct Performance Benchmark Comparison Against Standard ML Frameworks
* Serialization and Archiving of the Trained Neural Network Object Artifacts

---

## Technical Context and Overview

### What I Did on Day 17
On Day 17, I trained a lightweight Multi-Layer Perceptron (MLP) neural network using a compact `(8, 4)` hidden layer design tailored to match memory constraints of embedded hardware systems. The network was evaluated against the standardized validation metrics using our master sheet splits.

---

## Cross-Model Performance Benchmarking Matrix

| Algorithm Track Evaluated | Key Hyperparameters Selected | Accuracy | F1-Score | Bit Error Rate (BER) | Resource Complexity |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Fixed Threshold** | Midpoint Boundary = 0.5 | 1.0000 | 1.0000 | 0.0000e+00 | Minimal (O(1)) |
| **Support Vector Machine** | Kernel=RBF, C=1.0 | 1.0000 | 1.0000 | 0.0000e+00 | Moderate |
| **Random Forest** | Trees=50, Max Depth=3 | 1.0000 | 1.0000 | 0.0000e+00 | Moderate |
| **Multi-Layer Perceptron** | Hidden Layers=(8, 4) | 1.0000 | 1.0000 | 0.0000e+00 | High (Matrix Mult) |
