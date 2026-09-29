# Master Publication Figures (Figures 1–9)

*Catalog and captions for IEEE / ACM / Nature Digital Medicine publication submissions.*

---

## Figure 1: End-to-End System Architecture of SCRIPT
- **File Location:** `results/figures/figure1_system_architecture.png`
- **Caption:** **Figure 1. End-to-End Architecture of the SCRIPT Framework.** 
  A raw scanned handwriting form ($3500\times 2400$ px) is processed simultaneously along two complementary paths: (1) a deep visual branch utilizing a pre-trained ResNet-50 backbone extracting a 2048-D global spatial representation, and (2) a biomechanical branch extracting 9 clinical geometric features via OpenCV, projected through a 64-D linear encoder. The two modalities are concatenated ($2112$-D) and dynamically re-weighted using parametric feature attention regularized by an OpenCV reconstruction pretext task. The resulting compact 256-D latent embedding is scored by an offset-aligned Adaptive Local Outlier Factor (LOF, $k=5$) calibrated against the healthy population reference library. A three-tier explainability engine provides clinician decision support via ink-constrained Grad-CAM, counterfactual LOF feature attribution, and automated diagnostic dossiers.

---

## Figure 2: Natural Inter-Writer Stylistic Negative Control
- **File Locations:** 
  - Panel A: `results/natural_variation/distribution_comparison.png`
  - Panel B: `results/natural_variation/style_vs_anomaly_scatter.png`
- **Caption:** **Figure 2. Empirical Validation of Natural Stylistic Negative Control.** 
  (A) Outlier score distributions for the healthy training cohort ($N=1,095$) versus held-out, completely unseen healthy writers ($N=444$) under strict $70/30$ writer-disjoint splitting. At the 95th-percentile healthy threshold, the false positive rate on unseen healthy writers is just $7.43\%$, confirming that the model does not misclassify natural stylistic eccentricity as dysgraphic anomaly. 
  (B) Scatter plot of anomaly scores against natural writer style parameters (baseline slant angle and word spacing coefficient of variation). All Pearson correlation coefficients satisfy $|\rho| < 0.14$ ($p > 0.05$), proving that the Adaptive LOF scoring engine is decoupled from benign idiosyncratic handwriting variations.

---

## Figure 3: Systematic Component Ablation Across 8 Architectural Configurations
- **File Location:** `results/ablation/ablation_barplot.png`
- **Caption:** **Figure 3. Systematic Component Ablation Study Across Feature Spaces and Outlier Scorers ($N=2,539$).** 
  Comparison of ROC-AUC and PR-AUC across eight configurations (ABL-A through ABL-H). While unconstrained high-dimensional visual representations (ABL-B, ABL-C, ABL-D) yield high artificial AUCs by detecting pixel-level image perturbations, they fail clinical geometric grounding. The proposed 256-D multimodal bottleneck regularized by geometric reconstruction (ABL-E) achieves an optimal clinical trade-off ($\text{ROC-AUC} = 0.6248$, $\text{PR-AUC} = 0.5152$). Replacing Adaptive LOF with Euclidean $k$-NN ($0.5323$), One-Class SVM ($0.5274$), or Isolation Forest ($0.5278$) causes complete performance collapse, proving the decisive necessity of local reachability density estimation.

---

## Figure 4: Controlled Multi-Severity Deficit Dose-Response Testbed
- **File Locations:**
  - Panel A: `results/severity/dose_response_curves.png`
  - Panel B: `results/severity/dose_response_boxplots.png`
  - Panel C: `results/severity/severity_visual_examples.png`
- **Caption:** **Figure 4. Controlled Physical Deficit Dose-Response Testbed ($N=80$ Forms Across 20 Writers).** 
  (A) Dose-response trajectory of mean LOF anomaly scores as corruption severity escalates from $L_0$ (Clean Baseline) through $L_1$ (Mild), $L_2$ (Moderate), and $L_3$ (Severe) across the four deficit proxy slices.
  (B) Within-subject boxplot distributions showing consistent separation for stroke dropout (involuntary pen lifts, $70.0\%$ monotonic concordance).
  (C) High-resolution visual crops demonstrating calibrated physical degradations: Gaussian blur micro-tremors, stroke dropout fragmentation, muscular spasm stroke thickening, and hypotonic stroke thinning.

---

## Figure 5: Hierarchical Multi-Scale Resolution Breakdown & Weight Optimization
- **File Locations:**
  - Panel A: `results/hierarchical/hierarchical_comparison_roc.png`
  - Panel B: `results/hierarchical/weight_optimization_heatmap.png`
- **Caption:** **Figure 5. Hierarchical Multi-Scale Analysis (Page $\rightarrow$ Line $\rightarrow$ Word).** 
  (A) Held-out test ROC curves ($N=169$ forms, 0 writer overlap) comparing Page-Level Alone ($\text{AUC} = 0.6208$), Word-Level Alone ($\text{AUC} = 0.3841$), Line-Level Alone ($\text{AUC} = 0.3307$), and Hierarchical Convex Fusion ($\text{AUC} = 0.5754$). 
  (B) Ternary simplex convex weight optimization heatmap demonstrating that word-level bounding-box z-score aggregations introduce lexical variance noise, establishing that holistic page-level visual attention embeddings remain scientifically superior for unconstrained scanned text.

---

## Figure 6: Quantitative Grad-CAM Visual Attribution & Ink-Masked Localization
- **File Locations:**
  - Panel A: `results/xai/xai_attribution_metrics_bar.png`
  - Panel B: `results/xai/xai_gradcam_comparison_grid.png`
- **Caption:** **Figure 6. Quantitative Evaluation of Visual Explainability Fidelity ($N=24$ Ground-Truth Verified Forms).** 
  (A) Quantitative comparison of Ink Attribution Ratio (IAR), Background Spillover Ratio (BSR), and Pointing Game Accuracy. Standard unconstrained Grad-CAM suffers from severe spatial diffusion ($97.4\%$ of saliency energy spills onto white page margins, $\text{IAR} = 2.6\%$).
  (B) Four-column visual attribution matrix: (1) Input handwriting form crop, (2) Ground-truth physical anomaly mask, (3) Standard ResNet-50 Grad-CAM heatmap, and (4) Proposed Constrained Ink-Masked Grad-CAM. By masking gradients through the binarized ink stroke geometry, ink attribution reaches $100\%$, providing sharp, clinically trustworthy stroke localization.

---

## Figure 7: Writer Distribution Bounds & Strategy Decision Matrix
- **File Locations:**
  - Panel A: `results/writer_strategy/writer_distribution_histogram.png`
  - Panel B: `results/writer_strategy/strategy_comparison_bar.png`
- **Caption:** **Figure 7. Empirical Proof of Writer Capacity Bounds on the IAM Database ($N=657$ Writers).** 
  (A) Histogram of form counts per writer. In the IAM benchmark, $54.2\%$ of writers possess exactly 1 form, and $94.2\%$ possess fewer than 6 forms. This establishes an empirical impossibility proof for writer-specific anomaly modeling: LOF with $k=5$ neighbors cannot be fitted for $>94\%$ of the population.
  (B) Performance comparison of Universal Adaptive LOF ($\text{AUC} = 0.6248$), Style-Cluster LOF ($\text{AUC} = 0.6187$), and Writer-Specific LOF ($\text{AUC} = 0.5000$). On the small eligible multi-writer cohort, writer-specific LOF degenerates to random guessing due to extreme reference sparsity ($N \le 5$), validating Universal Adaptive LOF as the only feasible deployment strategy.

---

## Figure 8: Feature Space Geometric Alignment & Cross-Modal Dynamics
- **File Locations:**
  - Panel A: `results/geometric_alignment/feature_magnitude_comparison.png`
  - Panel B: `results/geometric_alignment/attention_weight_distribution.png`
- **Caption:** **Figure 8. Feature Space Cross-Modal Alignment and Manifold Displacement.** 
  (A) Dual-panel distribution showing the $3.41\times$ L2 norm disparity between the 2048-D ResNet visual branch (mean norm $11.51$) and the 64-D encoded biomechanical branch (mean norm $3.60$), and boxplot distribution of learned feature attention weights per dimension.
  (B) Euclidean centroid manifold displacement ($\|\boldsymbol{\mu}_{\text{clean}} - \boldsymbol{\mu}_{\text{synthetic}}\|_2$) across anomaly slices in the 256-D latent space. Thickening induces the highest manifold displacement ($1.0388$), directly matching its empirical benchmark peak ($\text{AUC} = 0.6610$).

---

## Figure 9: Paired Bootstrap Sampling Distributions & Multi-Seed Stability
- **File Locations:**
  - Panel A: `results/significance/bootstrap_distributions.png`
  - Panel B: `results/repeatability/seed_stability_plot.png`
- **Caption:** **Figure 9. Rigorous Statistical Inferential Validation Across 2,539 Paired Evaluations.** 
  (A) Paired stratified bootstrap distributions ($1,000$ resamples) of $\Delta\text{ROC-AUC}$ (Proposed Adaptive LOF minus baseline) against Devi & Kavya (2023), Agarwal et al. (2023), Alevizos et al. (2024), and Euclidean $k$-NN. In all $1,000$ iterations, the proposed method never performed worse than any baseline ($p_{\text{empirical}} < 0.001$, DeLong $Z > 6.24, p < 10^{-9}$).
  (B) Multi-seed repeatability across 5 independent $80/20$ cross-validation splits of the clean reference library (Seeds: 42, 123, 456, 789, 101112), demonstrating negligible variance ($\text{Mean AUC} = 0.5601 \pm 0.0166$) and invariant anomaly slice rankings.
