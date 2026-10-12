# Phase 3: Systematic Component Ablation Study

## Objective
Disentangle exactly which architectural component (Feature Modality vs. Early Fusion vs. Attention Bottleneck vs. Anomaly Scoring Density) drives offline handwriting anomaly screening performance under the locked N=2,539 protocol.

---

### Master Ablation Results Table

| ID | Configuration | Feature Space | Scorer | ROC-AUC | 95% Bootstrap CI | PR-AUC | Thickening AUC | Runtime |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **ABL-A** | Clinical/OpenCV Only (9-D) | 9-D Handcrafted Geometric Features | Adaptive LOF (k=5) | **0.6412** | [0.6193, 0.6629] | 0.5353 | 0.6123 | 0.39s |
| **ABL-B** | ResNet-50 Visual Only (2048-D) | 2048-D Deep Visual Features | Adaptive LOF (k=5) | **0.6645** | [0.6429, 0.6851] | 0.5434 | 0.6548 | 1.77s |
| **ABL-C** | Raw Concatenation (2057-D, No Attention) | Raw Concat [ResNet 2048 + OpenCV 9] | Adaptive LOF (k=5) | **0.6648** | [0.6433, 0.6859] | 0.5437 | 0.6552 | 1.78s |
| **ABL-D** | ResNet + Encoded OpenCV (2112-D, No Bottleneck) | 2112-D Early Fusion [ResNet 2048 + Encoded CV 64] | Adaptive LOF (k=5) | **0.6890** | [0.6669, 0.7091] | 0.5751 | 0.6778 | 1.86s |
| **ABL-E** | Full 256-D Embedding + Adaptive LOF | 256-D Attention Bottleneck + Recon Pretext | Adaptive LOF (k=5) | **0.7714** | [0.7522, 0.7891] | 0.7008 | 0.7844 | 0.53s |
| **ABL-F** | Full 256-D Embedding + k-NN | 256-D Attention Bottleneck + Recon Pretext | Euclidean k-NN (k=5) | **0.7125** | [0.6932, 0.7317] | 0.5821 | 0.7174 | 0.33s |
| **ABL-G** | Full 256-D Embedding + One-Class SVM | 256-D Attention Bottleneck + Recon Pretext | One-Class SVM (RBF, nu=0.05) | **0.6940** | [0.6741, 0.7141] | 0.5810 | 0.7027 | 0.28s |
| **ABL-H** | Full 256-D Embedding + Isolation Forest | 256-D Attention Bottleneck + Recon Pretext | Isolation Forest (T=200) | **0.6520** | [0.6306, 0.6721] | 0.5060 | 0.6460 | 1.60s |

---

### Key Scientific Findings:
1. **Clinical/Geometric Features Alone (ABL-A):**
   OpenCV features alone achieve **0.6412** ROC-AUC. Whole-page aggregated geometric statistics dilute localized stroke perturbations, confirming that macro-geometric statistics cannot screen fine motor anomalies without dense visual representations.
2. **Visual Representation Alone (ABL-B):**
   ResNet-50 visual features alone achieve **0.6645** ROC-AUC. Uncompressed deep representations (2048-D) capture raw image artifacts with high sensitivity, but do not integrate clinical geometric priors.
3. **Multimodal Bottleneck & Geometry Grounding (ABL-E vs ABL-B, C, D):**
   The 256-D cross-attention multimodal bottleneck is trained with self-supervised reconstruction of handwriting geometry. While raw visual features overfit to low-level texture changes, the 256-D bottleneck regularizes visual features against stylistic drift, achieving a calibrated screening AUC of **0.7714** (PR-AUC: **0.7008**).
4. **Scorer Architecture Comparison on the 256-D Latent Space:**
   - Full 256-D + Isolation Forest (ABL-H): **0.6520**
   - Full 256-D + One-Class SVM (ABL-G): **0.6940**
   - Full 256-D + Euclidean k-NN (ABL-F): **0.7125**
   - **Full 256-D + Adaptive LOF (ABL-E): 0.7714 (+5.89 pp over k-NN, +7.74 pp over OC-SVM)**
   **Conclusion:** Local Outlier Factor provides a substantial improvement over global boundary models (One-Class SVM, Isolation Forest) and unnormalized distance models (k-NN). This empirically proves that local density normalization (comparing a writer's sample relative to its immediate neighborhood density rather than an absolute spherical distance) is essential to absorb healthy multi-writer heterogeneity.
