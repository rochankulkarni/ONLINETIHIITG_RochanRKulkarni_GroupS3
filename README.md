# TIH-Internship-UWOC
Supervised bit classification for underwater optical communication — TIH IIT Guwahati AI/ML internship.
# Supervised Bit Classification for Underwater Optical Communication

**Programme:** TIH IIT Guwahati — AI/ML Online Internship (4 Weeks)
**Domain:** Machine Learning / Underwater Optical Communication / Binary Classification
**Faculty:** Dr. AVR Murthy, Applied Physics, DIAT (Research Lab: OCBL)

---

## Problem Statement

In underwater optical wireless communication, water turbidity changes the received optical signal amplitude, which can make a fixed threshold unreliable for recovering transmitted bits. This project builds supervised classifiers that recover the transmitted bit 0 or 1 robustly across varying NTU (turbidity) conditions, using a labelled dataset of transmitted bits, received analog signal values, and turbidity levels.

## Objectives

- Understand the domain problem and formulate it as binary classification of the transmitted bit.
- Inspect and prepare the supplied data (received analog values + turbidity/NTU info).
- Build a reproducible preprocessing and exploratory-analysis pipeline.
- Implement and compare: fixed-threshold baseline, Logistic Regression, SVM, Random Forest, and an optional small MLP.
- Evaluate models using accuracy, precision, recall, F1, confusion matrix, BER/error rate, and inference time.
- Analyze robustness across individual NTU levels and compare threshold detection against ML classifiers.

## Repository Structure (planned)

```
├── data/               # Raw and processed datasets (raw copy kept unchanged)
├── notebooks/          # EDA, preprocessing, and modelling notebooks
├── src/                # Reusable preprocessing/modelling scripts
├── reports/            # Literature review, analysis write-ups, experiment logs
├── results/            # Metrics, plots, confusion matrices
└── README.md
```


## Tools & Libraries

Python, Jupyter/Google Colab, NumPy, Pandas, Matplotlib/Pyplot, Scikit-learn, seaborn.

## Notes

This is an individually executed project - all code, analysis, and results in this repository are independently produced as part of the internship's compulsory workflow.
