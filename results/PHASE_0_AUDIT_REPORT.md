# SCRIPT: Phase 0 Rigorous Codebase & Pipeline Audit Report

**Date:** 2026-09-25  
**Audit Scope:** Full codebase (`src/model.py`, `src/train.py`, `src/evaluate.py`, `src/word_anomaly.py`, `src/word_gradcam.py`, `src/form_feature_attribution.py`, `src/create_final_report.py`, `src/benchmark_literature_baselines.py`, metadata, synthetic generators, saved checkpoints).

---

## 1. Executive Audit Summary

| Audit Item | Status | Key Finding | Action / Implication for Research |
|:---|:---:|:---|:---|
| **1. Dataset Splitting** | ⚠️ Warning | Random index split in `train.py` creates massive writer overlap. | Phase 2 & 8 must enforce strict writer-independent splits. |
| **2. Writer Leakage** | ⚠️ Severe | **68.1% of validation writers** also appear in training data. | The encoder learned with overlapping writer identities. |
| **3. Normalization Leakage** | ✅ Passed | Computed strictly on training indices and stored in checkpoint. | Clean; no normalization leakage into validation. |
| **4. LOF Reference Purity** | ✅ Passed | Evaluated via strict Leave-One-Out (transductive LOF) on clean. | Clean reference evaluation is mathematically sound. |
| **5. Writer Distribution** | ℹ️ Critical Info | 54.2% of writers (356/657) have only 1 form; median = 1.0. | Writer-specific LOF is infeasible for >50% of the dataset. |
| **6. Synthetic Generation** | ✅ Verified | 1,000 forms generated from 742 clean forms across 4 slices. | XML-guided local ink perturbations verified. |
| **7. Clean Source Pairing** | ℹ️ Verified | Synthetic forms have their clean sources in the clean reference. | Makes anomaly detection harder, not easier (anchoring). |
| **8. Hyperparameter Tuning** | ⚠️ Caution | $k \in [1..50]$ was explored on the benchmark test set. | Full parameter sensitivity curves must be reported. |
| **9. Metric Reproducibility** | ✅ 100% Verified | ROC-AUC = **0.6248**, PR-AUC = **0.5152** reproduced to 4 decimals. | Baseline locked without discrepancies. |

---

## 2. In-Depth Audit Findings by Item

### Item 1 & 2: Dataset Splitting & Writer Identity Leakage
- **Observation:** `src/train.py` (lines 88–103) uses PyTorch's `random_split(dataset, [1232, 307], generator=seed(42))` to create the 80% train and 20% validation split.
- **Empirical Audit:**
  * Total clean forms: 1,539.
  * Training forms: 1,232 (comprising 583 unique writers).
  * Validation forms: 307 (comprising 232 unique writers).
  * **Writer Overlap:** 158 writers appear in **both** training and validation splits.
  * **Leakage Percentage:** $\frac{158}{232} = \mathbf{68.10\%}$ of validation writers were seen during self-supervised reconstruction training.
- **Scientific Impact:** For the self-supervised reconstruction pretext task, this means the feature attention network has seen stylistic features of 68.1% of validation writers. In our Phase 2 negative control and Phase 8 investigations, we will construct a strict writer-disjoint split.

### Item 3: Feature Normalization Rigor
- **Observation:** `train.py` (lines 110–126) calculates `feature_mean` and `feature_std` of the 9 OpenCV features strictly on `train_indices`.
- **Validation:** When `evaluate.py` and `save_paired_embeddings.py` run, they load these exact statistics from `best_dysgraphia_encoder.pth`. Test samples never contribute to the mean/std parameters.
- **Conclusion:** **No feature normalization leakage.**

### Item 4: LOF Reference Bank Integrity
- **Clean Forms Scoring:** In `benchmark_literature_baselines.py`, transductive `LocalOutlierFactor(n_neighbors=5, novelty=False)` is used. The anomaly score is computed from `-negative_outlier_factor_`, which scikit-learn computes using an exact Leave-One-Out (LOO) formulation:
  $$\text{LOF}(p) = \frac{1}{|N_k(p)|} \sum_{o \in N_k(p)} \frac{\text{lrd}(o)}{\text{lrd}(p)}$$
  where $N_k(p)$ contains only the $k$ neighbors excluding $p$ itself.
- **Synthetic Forms Scoring:** Evaluated via inductive `LocalOutlierFactor(novelty=True).decision_function(synth_fused)`. No synthetic sample was present during fitting.
- **Conclusion:** **Mathematically valid one-class protocol.**

### Item 5: Distribution of Writers Across the 1,539 Forms
Analysis of `data/metadata/metadata_all.csv`:
- Total forms: 1,539
- Total distinct writers: 657
- Summary statistics:
  * Minimum: 1 form per writer
  * Maximum: 59 forms per writer (writer `000` has 59 forms)
  * Mean: 2.342 forms per writer
  * Median: 1.0 form per writer
  * Standard Deviation: 3.033 forms
- **Histogram of Writer Capacity:**
  * Exactly 1 form: 356 writers (54.19%)
  * Exactly 2 forms: 142 writers (21.61%)
  * Exactly 3 forms: 32 writers (4.87%)
  * 4 or more forms: 127 writers (19.33%)
- **Critical Implication:** A "writer-specific" anomaly detection model (where an individual child's writing is scored only against their own previous clean samples) is statistically impossible for $>54\%$ of the IAM dataset. A population-based local density approach (like Adaptive LOF) is structurally required.

### Item 6 & 7: Synthetic Generation & Source Form Pairing
- **Generation Logic:** `src/synthetic_form_anomaly_generator.py` uses IAM XML word bounding boxes to build an exact ink mask, applying local morphology (`cv2.erode`, `cv2.dilate`), Gaussian blur, or dropout.
- **Source Form Distribution:** 1,000 synthetic forms were generated from 742 unique clean source forms.
- **Effect on Benchmark:** Because each synthetic form's original clean counterpart is present in the 1,539 clean reference library:
  * A synthetic form's closest neighbor in the clean reference library is almost always its own clean twin.
  * If a synthetic degradation is mild, its Euclidean distance to its clean twin remains small.
  * This anchor makes the synthetic form look *more normal* to the reference library, thereby **depressing** the anomaly score and making detection **harder**, not easier.
  * This confirms that the 0.6248 AUC is not an inflated artifact of pairing.

### Item 8: Hyperparameter Selection Audit
- In `src/test_knn_k.py`, $k \in \{1, 3, 5, 10, 20, 50\}$ was evaluated against `paired_embeddings.npz`.
- While $k=5$ was selected based on balancing noise suppression against cluster dilution, evaluating on the full test set represents an unblinded exploration.
- **Remedy:** In Phase 12, hyperparameter sensitivity will be formally evaluated on a dedicated validation split, and the entire sensitivity curve will be reported.

### Item 9: Exact Metric Reproducibility
Executing `src/benchmark_literature_baselines.py` under Python 3.12 with seed 42 produced:

| Method | Target Documented AUC | Reproduced AUC | Target Documented AP | Reproduced AP | Match? |
|:---|:---:|:---:|:---:|:---:|:---:|
| Devi & Kavya (2023) | 0.5276 | 0.527568 | 0.4216 | 0.421596 | ✅ Exact |
| Agarwal et al. (2023) | 0.5051 | 0.505149 | 0.3998 | 0.399777 | ✅ Exact |
| Alevizos et al. (2024) | 0.5123 | 0.512323 | 0.4051 | 0.405056 | ✅ Exact |
| Proposed (k-NN Ablation) | 0.5323 | 0.532299 | 0.4340 | 0.434030 | ✅ Exact |
| **Proposed (Adaptive LOF)** | **0.6248** | **0.624793** | **0.5152** | **0.515224** | ✅ Exact |

**Per-Slice Breakdown for Proposed Adaptive LOF:**
- Blur AUC: **0.6030** (Target: 0.6030)
- Dropout AUC: **0.6302** (Target: 0.6302)
- Thickening AUC: **0.6610** (Target: 0.6610)
- Thinning AUC: **0.6041** (Target: 0.6041)

---

## 3. Recommendations & Roadmap for Phases 1–17

1. **Phase 1 (Done):** Lock baseline predictions, metrics, and configurations in `results/baseline/`.
2. **Phase 2 (Negative Control):** Test whether natural writer style differences (slant, density, spacing) trigger false positive anomalies on clean negative controls.
3. **Phase 3 (Ablation):** Disentangle the exact performance contribution of:
   - OpenCV features alone
   - ResNet visual features alone
   - Concatenation without attention
   - Feature attention
   - Self-supervised reconstruction pretext loss
   - Scorer choice (k-NN, OC-SVM, Isolation Forest, Adaptive LOF)
4. **Phase 4 & 5 (Severity Analysis):** Introduce mathematically graduated severity levels (Mild, Moderate, Strong) and verify monotonic score responses.
5. **Phase 6 (Hierarchical Fusion):** Explore whether XML-derived page $\rightarrow$ line $\rightarrow$ word multi-scale integration improves discriminability.
6. **Phase 7 (Quantitative Grad-CAM):** Compute the Ink Attribution Ratio ($\text{IAR}$) to verify that heatmaps concentrate on actual ink pixels.
7. **Phase 10 (Statistical Significance):** Compute 95% bootstrap confidence intervals and DeLong tests for all AUC comparisons.

*Phase 0 Audit Completed & Verified.*
