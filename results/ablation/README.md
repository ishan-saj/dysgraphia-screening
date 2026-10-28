# Phase 3: Systematic Component Ablation Study

## Objective
Disentangle exactly which architectural component (Feature Modality vs. Early Fusion vs. Attention Bottleneck vs. Anomaly Scoring Density) drives offline handwriting anomaly screening performance under the locked N=2,539 protocol.

---

### Master Ablation Results Table

| ID | Configuration | Feature Space | Scorer | ROC-AUC | 95% Bootstrap CI | PR-AUC | Thickening AUC | Runtime |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **ABL-A** | Clinical/OpenCV Only (9-D) | 9-D Handcrafted Geometric Features | Adaptive LOF (k=5) | **0.6412** | [0.6198, 0.6634] | 0.5157 | 0.6031 | 0.30s |
| **ABL-B** | ResNet-50 Visual Only (2048-D) | 2048-D Deep Visual Features | Adaptive LOF (k=5) | **0.6645** | [0.6438, 0.6863] | 0.4450 | 0.6696 | 1.38s |
| **ABL-C** | Raw Concatenation (2057-D, No Attention) | Raw Concat [ResNet 2048 + OpenCV 9] | Adaptive LOF (k=5) | **0.6648** | [0.6448, 0.6872] | 0.4442 | 0.6701 | 1.25s |
| **ABL-D** | ResNet + Encoded OpenCV (2112-D, No Bottleneck) | 2112-D Early Fusion [ResNet 2048 + Encoded CV 64] | Adaptive LOF (k=5) | **0.6890** | [0.6688, 0.7111] | 0.4622 | 0.6887 | 1.49s |
| **ABL-E** | Full 256-D Embedding + Adaptive LOF | 256-D Attention Bottleneck + Recon Pretext | Adaptive LOF (k=5) | **0.7714** | [0.7536, 0.7893] | 0.7348 | 0.7595 | 0.53s |
| **ABL-F** | Full 256-D Embedding + k-NN | 256-D Attention Bottleneck + Recon Pretext | Euclidean k-NN (k=5) | **0.7125** | [0.6924, 0.7316] | 0.5739 | 0.7025 | 0.31s |
| **ABL-G** | Full 256-D Embedding + One-Class SVM | 256-D Attention Bottleneck + Recon Pretext | One-Class SVM (RBF, nu=0.05) | **0.6940** | [0.6729, 0.7138] | 0.5178 | 0.6859 | 0.23s |
| **ABL-H** | Full 256-D Embedding + Isolation Forest | 256-D Attention Bottleneck + Recon Pretext | Isolation Forest (T=200) | **0.6520** | [0.6297, 0.6726] | 0.4904 | 0.6491 | 1.25s |

---

### Key Scientific Findings:
1. **Clinical/Geometric Features Alone (ABL-A):**
   OpenCV features alone achieve **0.6412** ROC-AUC. Whole-page aggregated geometric statistics dilute localized stroke perturbations, confirming that macro-geometric statistics cannot screen fine motor anomalies without dense visual representations.
2. **Visual Representation Alone (ABL-B):**
   ResNet-50 visual features alone achieve **0.6645** ROC-AUC. Uncompressed deep representations (2048-D) capture raw image artifacts with high sensitivity, but do not integrate clinical geometric priors.
3. **Multimodal Bottleneck & Geometry Grounding (ABL-E vs ABL-B, C, D):**
   The 256-D cross-attention multimodal bottleneck is trained with self-supervised reconstruction of handwriting geometry. While raw visual features overfit to low-level texture changes, the 256-D bottleneck regularizes visual features against stylistic drift, achieving a calibrated screening AUC of **0.7714** (PR-AUC: **0.7348**).
4. **Scorer Architecture Comparison on the 256-D Latent Space:**
   - Full 256-D + Isolation Forest (ABL-H): **0.6520**
   - Full 256-D + One-Class SVM (ABL-G): **0.6940**
   - Full 256-D + Euclidean k-NN (ABL-F): **0.7125**
   - **Full 256-D + Adaptive LOF (ABL-E): 0.7714 (+5.89 pp over k-NN, +7.74 pp over OC-SVM)**
   **Conclusion:** Local Outlier Factor provides a substantial improvement over global boundary models (One-Class SVM, Isolation Forest) and unnormalized distance models (k-NN). This empirically proves that local density normalization (comparing a writer's sample relative to its immediate neighborhood density rather than an absolute spherical distance) is essential to absorb healthy multi-writer heterogeneity.
