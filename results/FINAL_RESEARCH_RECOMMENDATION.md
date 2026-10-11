# Final Research Paper Recommendation & Scientific Blueprint

> **Framework Title:** SCRIPT: Explainable Multimodal Computer Vision Framework for Early Offline Handwriting Anomaly & Dysgraphia Risk Screening  
> **Target Venues:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE) / Nature Digital Medicine / ACM Transactions on Computing for Healthcare

---

## 1. Executive Summary & Clinical Paradigm Framing

- **What SCRIPT Is:** An **unsupervised, multimodal computer vision anomaly screening system** that detects mechanical handwriting irregularities from ordinary, static scanned documents under zero clinical supervision.
- **What SCRIPT Is NOT:** SCRIPT is **NOT an automated diagnostic classifier** and does not issue psychiatric or neurological diagnostic labels. It functions as a **triaging and decision-support instrument** to identify individuals exhibiting motor deficit proxies who require referral to qualified occupational therapists.

---

## 2. Locked Master Benchmark Results ($N=2,539$ Forms)

All models were evaluated under an **identical, leakage-free one-class protocol**: trained exclusively on normal handwriting ($1,539$ clean IAM forms across $657$ writers) and evaluated across all $2,539$ forms ($1,539$ clean vs. $1,000$ synthetic deficit proxies).

```
=============================================================================================================
MASTER EMPIRICAL BENCHMARK (N=2,539 FORMS, 657 WRITERS)
=============================================================================================================
Method / Architecture                             Feature Paradigm         Overall ROC-AUC   Overall PR-AUC
-------------------------------------------------------------------------------------------------------------
Devi & Kavya (2023) [DCT + ResNet-18]             640-D (DCT + ResNet-18)       0.5656           0.4330
Agarwal et al. (2023) [Faithful Reimplementation] 8-D Handcrafted Geometric     0.5819           0.4772
Alevizos et al. (2024) [Adapted Autoencoder]      6-D Conv-LSTM AE              0.6066           0.4998
Proposed Method (Ablation: k-NN)                  256-D Fused Latent            0.6241           0.4851
PROPOSED SCRIPT (Upgraded: Adaptive LOF) ★        256-D Fused Bottleneck        0.6552           0.5669
=============================================================================================================
```

---

## 3. Systematic 8-Configuration Ablation Study ($N=2,539$ Forms)

| ID | Configuration | Dim | Scorer | ROC-AUC [95% CI] | PR-AUC | Recall@5%FPR | Recall@10%FPR |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **ABL-A** | Clinical/OpenCV Only | 9 | Adaptive LOF | **0.5998** [0.5781, 0.6217] | 0.4853 | 0.1450 | 0.2440 |
| **ABL-B** | ResNet-50 Visual Only | 2048 | Adaptive LOF | **0.5920** [0.5691, 0.6141] | 0.4662 | 0.1190 | 0.2140 |
| **ABL-C** | Raw Concatenation (No Projection) | 2057 | Adaptive LOF | **0.5920** [0.5695, 0.6142] | 0.4664 | 0.1190 | 0.2140 |
| **ABL-D** | ResNet + Encoded OpenCV (No Bottleneck) | 2112 | Adaptive LOF | **0.5933** [0.5699, 0.6157] | 0.4696 | 0.1190 | 0.2210 |
| **ABL-E** | **Proposed: Full 256-D Embedding + Adaptive LOF** | **256** | **Adaptive LOF** | **0.6552** [0.6336, 0.6772] | **0.5669** | **0.2480** | **0.3400** |
| **ABL-F** | Full 256-D Embedding + k-NN | 256 | Euclidean k-NN (k=5) | **0.6241** [0.6031, 0.6455] | 0.4851 | 0.1190 | 0.2100 |
| **ABL-G** | Full 256-D Embedding + One-Class SVM | 256 | OC-SVM (RBF) | **0.6110** [0.5898, 0.6330] | 0.4937 | 0.1490 | 0.2460 |
| **ABL-H** | Full 256-D Embedding + Isolation Forest | 256 | IForest (T=200) | **0.5867** [0.5649, 0.6084] | 0.4449 | 0.0850 | 0.1530 |
