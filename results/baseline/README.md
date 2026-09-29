# Locked Baseline Benchmark Results

## Status: LOCKED (Phase 1)
**Date Established:** 2026-09-25  
**Random Seed:** 42  
**Protocol:** One-class unsupervised anomaly detection (Reference Clean $N=1,539$, Synthetic $N=1,000$).

---

### Master Empirical Baseline Metrics

| Method | Overall ROC-AUC | Overall PR-AUC | Blur AUC | Dropout AUC | Thickening AUC | Thinning AUC | Sensitivity (@5% FPR) | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Devi & Kavya (2023)** | 0.5276 | 0.4216 | 0.5228 | 0.5540 | 0.5411 | 0.4950 | 7.9% | 0.1345 |
| **Agarwal et al. (2023)** | 0.5051 | 0.3998 | 0.4910 | 0.4914 | 0.5282 | 0.5078 | 7.5% | 0.1288 |
| **Alevizos et al. (2024)** | 0.5123 | 0.4051 | 0.5025 | 0.5149 | 0.5211 | 0.5098 | 7.2% | 0.1245 |
| **Proposed (k-NN Ablation)** | 0.5323 | 0.4340 | 0.5135 | 0.5370 | 0.5569 | 0.5206 | 8.8% | 0.1479 |
| **Proposed (Adaptive LOF) [Target Baseline]** | **0.6248** | **0.5152** | **0.6030** | **0.6302** | **0.6610** | **0.6041** | **17.9%** | **0.2642** |

---

### Files in this Directory:
- `metrics.csv`: Machine-readable master metrics and classification counts.
- `predictions.csv`: Individual sample anomaly scores across all 2,539 forms for every baseline.
- `configuration.json`: Complete specification of parameters, seeds, and dimensions.
- `README.md`: This reference specification.

*DO NOT OVERWRITE OR MODIFY THE FILES IN THIS DIRECTORY.*
