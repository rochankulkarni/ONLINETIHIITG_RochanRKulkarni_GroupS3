# Day 14: Peer Consultation and Week-2 Package Readiness

## Deliverables Summary
* Cross-Peer Comparison Matrix of EDA Insights
* Detailed Evaluation of Difficult Environmental Classes and Condition-Wise Noise
* Documentation of One Peer Insight and One Frozen Research Hypothesis
* Consolidated Week-2 Project Package Assembly Verification

---

## Technical Context and Overview

### What I Did on Day 14
On Day 14, I finalized the Week-2 progress package by conducting a peer evaluation session. Rather than exchanging codebase files, we directly benchmarked our analytical EDA discoveries, signal distributions, and water degradation thresholds using our new master framework headers. We mapped out the explicit behavioral differences of the optical photoresistor receiver under difficult transmission classes, logged actionable peer observations regarding frequency-dependent noise spikes, and locked in our core machine learning hypotheses for the upcoming Week-3 multi-variable model training phase.

---

## Peer EDA Comparison and Diagnostic Review

### 1. Feature Distribution and Class Spacing Comparisons
* **Baseline Clear Tracking (1-2 NTU):** Peer review confirms absolute bimodal tracking separation. Both analyses isolated identical resting voltage clusters across the new master sheets: the logic-low state rests tightly at ~0.48V, while the logic-high state anchors at ~0.68V.
* **Turbidity Attenuation Trends:** Both evaluations independently discovered that increasing turbidity parameters do not impact the logic-low baseline voltage. Instead, light scattering selectively dampens the logic-high state peak, compressing its amplitude and pushing it down toward the hardware midpoint line.

### 2. Analysis of Difficult Classes and Conditions
The core difficulty across both project tracks centers on the **5-6 NTU (Severe Turbidity)** environmental class. At this depth, severe forward scattering and path attenuation cause significant problems:
* **Amplitude Compression:** The logic-high voltage cluster degrades significantly, dropping from its clear-water 0.68V profile down below 0.46V.
* **Boundary Blur Zone:** This downwards shift forces the high-state cluster to overlap directly with the upper variance tail of the logic-low state, creating a critical blur zone where static hardware thresholds misclassify bits.

---

## Peer Insight and Research Hypothesis Log

### 1. Actionable Peer Insight
> **Insight:** *The transient voltage drops observed during signal level transitions are not random environmental anomalies. They are hardware switching spikes caused by the capacitive rise-and-fall response time of the photoresistor circuit when handling high-frequency square wave pulses.*
* **Strategic Integration:** Rather than discarding these noisy data rows, we will use the `Delta_Gradient` feature engineered on Day 13. By calculating the first-order difference between sequential samples, the machine learning model can treat these switching spikes as timing markers to identify symbol transitions instead of misclassifying them as data errors.

### 2. Frozen Modeling Hypothesis
> **Hypothesis:** *While a fixed, hardcoded threshold boundary will achieve >99% decoding accuracy under low-turbidity conditions (1-2 NTU), its performance will completely collapse once turbidity scales past 3-4 NTU due to selective amplitude attenuation. Conversely, an instance-based model leveraging the engineered temporal features (variance and gradients) will maintain stable classification metrics above 90% across all turbidity levels by dynamically mapping spatial density changes rather than relying on a static midpoint.*
