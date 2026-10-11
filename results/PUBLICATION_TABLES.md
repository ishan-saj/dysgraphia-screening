# Master Publication Tables (Tables 1–8)
*Comprehensive empirical reference tables for IEEE / ACM / Nature Digital Medicine publication.*

---

## Table 1: Benchmark Dataset Partitioning & Clinical Deficit Proxies

| Data Cohort / Proxy Slice | Form Count ($N$) | Distinct Writers ($W$) | Clinical Motor Deficit Proxy | Physical Perturbation Mechanism |
|:---|:---:|:---:|:---|:---|
| **Clean IAM Reference (Train)** | 1,232 | ~525 | Normal adult handwriting | None (natural baseline) |
| **Clean IAM Reference (Validation)** | 307 | ~132 | Normal adult handwriting | None (natural baseline) |
| **Total Clean Reference Library** | **1,539** | **657** | **Healthy Population Baseline** | **Uncorrupted scanner captures** |
| Synthetic Slice: Thickening | 253 | Paired from Clean | Excessive pen force / muscular spasms | Morphological dilation ($3\times 3$ to $5\times 5$) |
| Synthetic Slice: Thinning | 276 | Paired from Clean | Faint grip / motor fatigue / paresis | Morphological erosion ($3\times 3$) |
| Synthetic Slice: Blur | 221 | Paired from Clean | Micro-tremor / cerebellar ataxia | Gaussian smoothing ($\sigma \in [1.2, 2.5]$) |
| Synthetic Slice: Dropout | 250 | Paired from Clean | Involuntary pen lifts / stroke fragmentation | Random patch masking ($p \in [0.08, 0.22]$) |
| **Total Synthetic Evaluation Cohort** | **1,000** | **Paired** | **Four Balanced Deficit Proxies** | **Deterministic physical operations** |
| **Total Benchmark Universe** | **2,539** | **657** | **Zero Data Leakage / One-Class Evaluation** | **1,539 Clean vs. 1,000 Corrupted** |

## Table 2: Master Empirical Benchmark & Literature Comparison

| Architecture / Published Model | Feature Paradigm ($D$) | Scoring Engine | Overall ROC-AUC | Overall PR-AUC | Blur AUC | Dropout AUC | Thickening AUC | Thinning AUC |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Devi & Kavya (2023) [DCT + ResNet-18]** | 640-D (DCT + ResNet-18) | Mahalanobis Distance | **0.5656** | **0.4330** | 0.5963 | 0.5162 | 0.5693 | 0.5822 |
| **Agarwal et al. (2023) [Faithful Reimplementation]** | 8-D Handcrafted Geometric | One-Class SVM (RBF) | **0.5819** | **0.4772** | 0.6265 | 0.5250 | 0.5950 | 0.5858 |
| **Alevizos et al. (2024) [Adapted Autoencoder]** | 6-D Geometric Sequence | Conv1D-LSTM Autoencoder MSE | **0.6066** | **0.4998** | 0.5934 | 0.6007 | 0.6305 | 0.6005 |
| **Proposed Method (Ablation: k-NN)** | 256-D Fused Bottleneck | Euclidean k-NN (k=5) | **0.6241** | **0.4851** | 0.6788 | 0.5462 | 0.6285 | 0.6468 |
| **Proposed Method (Upgraded: Adaptive LOF)** | 256-D Fused Bottleneck | Adaptive LOF | **0.6552** | **0.5669** | **0.7321** | **0.5743** | **0.6728** | **0.6508** |

## Table 3: Systematic Component Ablation Study across Feature & Scoring Paradigms

| Ablation ID | Model Architecture | Feature Representation ($D$) | Scoring Engine | Overall ROC-AUC | PR-AUC | 95% Confidence Interval | Scientific Finding / Conclusion |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---|
| **ABL-A** | Clinical/OpenCV Only (9-D) | 9-D Handcrafted Geometric Features | Adaptive LOF | **0.5998** | 0.4853 | [0.5781, 0.6217] | Geometric & intra-page features alone cannot capture full deep visual texture. |
| **ABL-B** | ResNet-50 Visual Only (2048-D) | 2048-D Deep Visual Features | Adaptive LOF | **0.5920** | 0.4662 | [0.5691, 0.6141] | High-dimensional visual features alone lack clinical geometric grounding. |
| **ABL-C** | Raw Concatenation (2057-D, No Attention) | Raw Concat [ResNet 2048 + OpenCV 9] | Adaptive LOF | **0.5920** | 0.4664 | [0.5695, 0.6142] | Unweighted concatenation is magnitude-dominated by 2048 visual dimensions. |
| **ABL-D** | ResNet + Encoded OpenCV (2112-D, No Bottleneck) | 2112-D Early Fusion [ResNet 2048 + Encoded CV 64] | Adaptive LOF | **0.5933** | 0.4696 | [0.5699, 0.6157] | Early concatenation without bottleneck suffers high-dimensional density dilution. |
| **ABL-E** | **Full 256-D Embedding + Adaptive LOF** | **256-D Attention Bottleneck + Recon Pretext** | **Adaptive LOF** | **0.6552** | **0.5669** | **[0.6336, 0.6772]** | **PROPOSED: Bottleneck regularized by reconstruction anchors multimodal representation.** |
| **ABL-F** | Full 256-D Embedding + k-NN | 256-D Attention Bottleneck + Recon Pretext | Euclidean k-NN (k=5) | **0.6241** | 0.4851 | [0.6031, 0.6455] | Unnormalized Euclidean distance lags local density ratio by -3.11 AUC points. |
| **ABL-G** | Full 256-D Embedding + One-Class SVM | 256-D Attention Bottleneck + Recon Pretext | One-Class SVM (RBF, nu=0.05) | **0.6110** | 0.4937 | [0.5898, 0.6330] | Global support vector boundary underperforms local density under style diversity. |
| **ABL-H** | Full 256-D Embedding + Isolation Forest | 256-D Attention Bottleneck + Recon Pretext | Isolation Forest (T=200) | **0.5867** | 0.4449 | [0.5649, 0.6084] | Orthogonal hyperplane partitioning struggles on dense 256-D handwriting manifolds. |