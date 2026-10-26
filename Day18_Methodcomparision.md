# Day 18: Model Comparison and Mentor Review Progress Pack

## Deliverables Summary
* Slide 1: Comprehensive Unified Comparison Matrix Across All Pipeline Methods
* Slide 2: System Constraints Review (Overfitting, Resource Complexity, Failure Patterns)
* Slide 3: Strategic Candidate Shortlist for Deeper Turbidity Experiments

---

## 📊 Slide 1: Comprehensive Unified Comparison Matrix

The table below logs the empirical performance scores evaluated on the isolated Dataset Version 1.0 test partition (derived exclusively from the clear-water `1-2NTU` master entries):

| Algorithmic Decoding Method | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TN, FP, FN, TP) | Bit Error Rate (BER) | Avg Single-Sample Latency (µs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Threshold (Control)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [8, 0, 0, 8] | 0.0000e+00 | 0.0210 µs |
| **Logistic Regression (ML)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [8, 0, 0, 8] | 0.0000e+00 | 0.1850 µs |
| **Support Vector Machine (RBF)**| 1.0000 | 1.0000 | 1.0000 | 1.0000 | [8, 0, 0, 8] | 0.0000e+00 | 2.4510 µs |
| **Random Forest (Depth=3)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [8, 0, 0, 8] | 0.0000e+00 | 4.1200 µs |
| **Multi-Layer Perceptron (8,4)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [8, 0, 0, 8] | 0.0000e+00 | 12.1540 µs |

---

## 📉 Slide 2: Structural Constraints and Risk Diagnosis

### 1. Overfitting Analysis and Data Limitations
Because the baseline `1-2NTU` tracking blocks within the master dataset contain a wide, unattenuated voltage gap between logic states, every single algorithmic method registers an absolute 100% success matrix. This serves as an invaluable structural validation of our code pipeline architecture. 

However, this represents an idealized data limitation. Presenting these perfect clear-water metrics as a final proof of system capability poses a severe overfitting risk. The models have only learned to separate columns under perfect transmission conditions and are currently unexposed to heavy light scattering.

### 2. Computational Complexity and Failure Patterns
* **Fixed Threshold Failure Pattern:** The rigid, non-learning midpoint decision line (0.5 scaled space) works well here but is fundamentally brittle. It cannot adapt to baseline drift or forward light scattering. Once turbidity increases, path loss will compress the logic-high peak downwards, causing massive false-negative errors.
* **Neural Network Complexity:** The Multi-Layer Perceptron (MLP) carries the heaviest resource penalty (12.1540 µs latency). Running continuous matrix multiplications across hidden neural layers introduces processing overhead that can limit transmission speeds on resource-constrained hardware like a Raspberry Pi receiver node.

---

## 🎯 Slide 3: Strategic Candidate Shortlist for Deeper Experiments

To prepare for cross-condition turbidity testing while balancing embedded hardware processing limits, the following two models are shortlisted for deeper experimentation:

### Candidate 1: Random Forest Ensemble (The Primary Robust Track)
* **Justification:** Random Forest achieved flawless telemetry scores while maintaining a highly efficient processing overhead (4.1200 µs). Its internal decision tree ensemble uses voting structures to judge continuous features, making it highly resilient to individual sensor fluctuations without requiring dense matrix computations.

### Candidate 2: Support Vector Machine with RBF Kernel (The Non-Linear Boundary Track)
* **Justification:** The SVM model scored perfect classification markers with a low single-sample inference latency of 2.4510 µs. Its Radial Basis Function (RBF) kernel projects input metrics into higher-dimensional structures, making it the strongest candidate to draw smooth, non-linear boundaries when high turbidity compresses our voltage ranges.

### Next Developmental Directive
The next phase will pass the high-turbidity master data blocks (`3-4NTU` and `5-6NTU`) straight through these two optimized candidates to evaluate their true channel resilience under severe scattering noise.
