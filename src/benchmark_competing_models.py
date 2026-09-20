#!/usr/bin/env python3
"""
Benchmark Competing Literature Paradigms and Proposed Methods on Synthetic Form Anomalies.

Evaluates 6 competing paradigms:
1. Baseline 1 (Asselborn et al., 2020 Paradigm):
   9 OpenCV geometric features -> PCA (3 components) -> Euclidean distance to clean normal centroid.
2. Baseline 2 (Devi & Kavya, 2023 / Pure Deep Visual Baseline):
   Pure ResNet-50 visual features (2048-D) -> k-NN scoring (k=5).
3. Baseline 3 (Agarwal et al., 2023 Paradigm):
   Hybrid 256-D fused features -> One-Class SVM (rbf kernel, nu=0.05).
4. Baseline 4 (Alevizos et al., 2024 Paradigm):
   Hybrid features -> Lightweight Autoencoder (256 -> 64 -> 16 -> 64 -> 256, MSELoss, Adam lr=1e-3,
   20 epochs, weight_decay=1e-4) -> Anomaly score = reconstruction error ||x - x_hat||^2.
5. Proposed Method (Ablation - Local Density):
   Hybrid 256-D fused features -> k-NN scoring (k=5).
6. Proposed Method (Upgraded - Adaptive Local Density):
   Hybrid 256-D fused features -> Local Outlier Factor (LOF with novelty=True, n_neighbors=15,
   metric='minkowski') -> Anomaly score = -decision_function(X_test).

Artifacts generated:
- Console Markdown comparison table
- CSV results: data/synthetic_form_anomalies/benchmark_model_comparison.csv
- Overlaid ROC curves: data/synthetic_form_anomalies/model_comparison_roc.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_FEATURE_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "resnet_features.pt"
)
METADATA_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"
)
PAIRED_CSV = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "paired_clean_vs_synthetic.csv"
)
FUSED_EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "paired_embeddings.npz"
)
SYNTHETIC_RESNET_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_resnet_features.pt"
)

ROC_PLOT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "model_comparison_roc.png"
)
OUTPUT_CSV_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "benchmark_model_comparison.csv"
)

# Known discrepancies in IAM form naming for metadata alignment
FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}


# ============================================================
# LIGHTWEIGHT AUTOENCODER (Alevizos et al., 2024 Paradigm)
# ============================================================

class LightweightAutoencoder(nn.Module):
    """
    Lightweight Autoencoder for tabular/embedding anomaly detection.
    Architecture: 256 -> 64 -> 16 -> 64 -> 256
    """
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ============================================================
# MAIN BENCHMARK FUNCTION
# ============================================================

def run_benchmark():
    # Set deterministic random seeds for full reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 85)
    print("BENCHMARKING COMPETING LITERATURE PARADIGMS & PROPOSED METHODS")
    print("=" * 85)

    # ------------------------------------------------------------
    # 1. Data Ingestion (Strictly Read-Only)
    # ------------------------------------------------------------
    print("\n[Step 1/5] Ingesting precomputed clean and synthetic features...")

    # Load clean features (OpenCV 9-D and ResNet 2048-D)
    assert CLEAN_FEATURE_FILE.exists(), f"Missing clean features file: {CLEAN_FEATURE_FILE}"
    clean_data = torch.load(
        CLEAN_FEATURE_FILE,
        map_location="cpu",
        weights_only=False,
    )
    clean_opencv = clean_data["opencv_features"].numpy()
    clean_resnet = clean_data["image_features"].numpy()
    clean_sample_ids = [str(x) for x in clean_data["sample_ids"]]

    # Load precomputed synthetic ResNet-50 visual features (2048-D)
    assert SYNTHETIC_RESNET_FILE.exists(), f"Missing synthetic ResNet features: {SYNTHETIC_RESNET_FILE}"
    synth_resnet_data = torch.load(
        SYNTHETIC_RESNET_FILE,
        map_location="cpu",
        weights_only=False,
    )
    synth_resnet = synth_resnet_data["image_features"].numpy()

    # Load hybrid fused 256-D embeddings
    assert FUSED_EMBEDDINGS_FILE.exists(), f"Missing fused embeddings file: {FUSED_EMBEDDINGS_FILE}"
    fused_data = np.load(
        FUSED_EMBEDDINGS_FILE,
        allow_pickle=True,
    )
    clean_fused = fused_data["clean_embeddings"]
    synth_fused = fused_data["synthetic_embeddings"]
    synth_types = np.array([str(x) for x in fused_data["synthetic_types"]])

    # Align paired clean OpenCV features for synthetic forms
    assert PAIRED_CSV.exists(), f"Missing paired CSV: {PAIRED_CSV}"
    assert METADATA_FILE.exists(), f"Missing metadata file: {METADATA_FILE}"
    paired_df = pd.read_csv(PAIRED_CSV)
    meta_df = pd.read_csv(METADATA_FILE)

    cv_lookup = {sid: clean_opencv[i] for i, sid in enumerate(clean_sample_ids)}
    form_to_sid = {
        str(r["form_id"]): str(r["sample_id"])
        for _, r in meta_df.iterrows()
    }

    synth_opencv_list = []
    for _, r in paired_df.iterrows():
        fid = str(r["clean_form_id"])
        fid = FORM_ID_MAP.get(fid, fid)
        synth_opencv_list.append(cv_lookup[form_to_sid[fid]])
    synth_opencv = np.array(synth_opencv_list, dtype=np.float32)

    # ------------------------------------------------------------
    # Robustness & Dimension Sanity Checks
    # ------------------------------------------------------------
    n_clean = len(clean_sample_ids)
    n_synth = len(synth_types)

    assert clean_opencv.shape == (n_clean, 9), f"clean_opencv shape mismatch: {clean_opencv.shape}"
    assert synth_opencv.shape == (n_synth, 9), f"synth_opencv shape mismatch: {synth_opencv.shape}"
    assert clean_resnet.shape == (n_clean, 2048), f"clean_resnet shape mismatch: {clean_resnet.shape}"
    assert synth_resnet.shape == (n_synth, 2048), f"synth_resnet shape mismatch: {synth_resnet.shape}"
    assert clean_fused.shape == (n_clean, 256), f"clean_fused shape mismatch: {clean_fused.shape}"
    assert synth_fused.shape == (n_synth, 256), f"synth_fused shape mismatch: {synth_fused.shape}"

    for arr_name, arr in [
        ("clean_opencv", clean_opencv),
        ("synth_opencv", synth_opencv),
        ("clean_resnet", clean_resnet),
        ("synth_resnet", synth_resnet),
        ("clean_fused", clean_fused),
        ("synth_fused", synth_fused),
    ]:
        assert not np.isnan(arr).any(), f"NaN detected in {arr_name}"
        assert not np.isinf(arr).any(), f"Inf detected in {arr_name}"

    print(f"  [OK] Clean reference samples (y=0):     {n_clean}")
    print(f"  [OK] Synthetic corrupted samples (y=1): {n_synth}")
    print(f"       - Blur:        {(synth_types == 'blur').sum()}")
    print(f"       - Dropout:     {(synth_types == 'dropout').sum()}")
    print(f"       - Thickening:  {(synth_types == 'thickening').sum()}")
    print(f"       - Thinning:    {(synth_types == 'thinning').sum()}")

    # ------------------------------------------------------------
    # 2. Fit and Score Competing Models
    # ------------------------------------------------------------
    print("\n[Step 2/5] Benchmarking models...")
    models = {}

    # Model 1: Baseline 1 (Asselborn et al., 2020 Paradigm)
    # 9 OpenCV geometric features -> PCA (3 components) -> Euclidean distance to clean normal centroid
    print("  -> (1/6) Baseline 1: Asselborn et al. (2020) [OpenCV (9-D) -> PCA(3) -> Centroid]...")
    scaler_cv = StandardScaler().fit(clean_opencv)
    clean_cv_scaled = scaler_cv.transform(clean_opencv)
    synth_cv_scaled = scaler_cv.transform(synth_opencv)

    pca_cv = PCA(n_components=3, random_state=42).fit(clean_cv_scaled)
    clean_pca = pca_cv.transform(clean_cv_scaled)
    synth_pca = pca_cv.transform(synth_cv_scaled)

    centroid = clean_pca.mean(axis=0)
    m1_clean_scores = np.linalg.norm(clean_pca - centroid, axis=1)
    m1_synth_scores = np.linalg.norm(synth_pca - centroid, axis=1)
    models["Baseline 1 (Asselborn et al., 2020) [OpenCV -> PCA(3) -> Centroid]"] = (
        m1_clean_scores,
        m1_synth_scores,
    )

    # Baseline 2 (Devi & Kavya / Deep Visual Baseline - SherlockNet style)
    # EXCLUDED per domain relevance instructions (outside handwriting dysgraphia screening domain)

    # Model 3: Baseline 3 (Agarwal et al., 2023 Paradigm)
    # Hybrid 256-D fused features -> One-Class SVM (rbf kernel, nu=0.05)
    print("  -> (3/6) Baseline 3: Agarwal et al. (2023) [Hybrid 256-D -> OCSVM (nu=0.05)]...")
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(clean_fused)
    m3_clean_scores = -ocsvm.decision_function(clean_fused)
    m3_synth_scores = -ocsvm.decision_function(synth_fused)
    models["Baseline 3 (Agarwal et al., 2023) [Hybrid 256-D -> OCSVM (nu=0.05)]"] = (
        m3_clean_scores,
        m3_synth_scores,
    )

    # Model 4: Baseline 4 (Alevizos et al., 2024 Paradigm)
    # Hybrid features -> Lightweight Autoencoder (256 -> 64 -> 16 -> 64 -> 256, MSELoss, Adam lr=1e-3, 20 epochs, weight_decay=1e-4)
    # Anomaly score = reconstruction error ||x - x_hat||^2
    print("  -> (4/6) Baseline 4: Alevizos et al. (2024) [Hybrid 256-D -> Autoencoder (20 eps)]...")
    ae = LightweightAutoencoder(input_dim=256)
    ae_criterion = nn.MSELoss()
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)

    clean_fused_tensor = torch.tensor(clean_fused, dtype=torch.float32)
    ae_train_loader = DataLoader(
        TensorDataset(clean_fused_tensor),
        batch_size=32,
        shuffle=True,
    )

    ae.train()
    for epoch in range(20):
        for batch_x, in ae_train_loader:
            ae_optimizer.zero_grad()
            recon_x = ae(batch_x)
            loss = ae_criterion(recon_x, batch_x)
            loss.backward()
            ae_optimizer.step()

    ae.eval()
    with torch.no_grad():
        synth_fused_tensor = torch.tensor(synth_fused, dtype=torch.float32)
        clean_recon = ae(clean_fused_tensor)
        synth_recon = ae(synth_fused_tensor)
        # ||x - x_hat||^2
        m4_clean_scores = torch.sum((clean_fused_tensor - clean_recon) ** 2, dim=1).numpy()
        m4_synth_scores = torch.sum((synth_fused_tensor - synth_recon) ** 2, dim=1).numpy()

    models["Baseline 4 (Alevizos et al., 2024) [Hybrid 256-D -> Autoencoder]"] = (
        m4_clean_scores,
        m4_synth_scores,
    )

    # Model 5: Proposed Method (Ablation - Local Density)
    # Hybrid 256-D fused features -> k-NN scoring (k=5)
    print("  -> (5/6) Proposed Method (Ablation): Local Density [Hybrid 256-D -> k-NN (k=5)]...")
    knn_fused = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn_fused.fit(clean_fused)
    clean_fused_dist, _ = knn_fused.kneighbors(clean_fused)
    synth_fused_dist, _ = knn_fused.kneighbors(synth_fused)
    m5_clean_scores = clean_fused_dist.mean(axis=1)
    m5_synth_scores = synth_fused_dist.mean(axis=1)
    models["Proposed Method (Ablation) [Hybrid 256-D -> k-NN (k=5)]"] = (
        m5_clean_scores,
        m5_synth_scores,
    )

    # Model 6: Proposed Method (Upgraded - Adaptive Local Density)
    # Hybrid 256-D fused features -> Local Outlier Factor (LOF with novelty=True, n_neighbors=15, metric='minkowski')
    # Anomaly score = -decision_function(X_test)
    print("  -> (6/6) Proposed Method (Upgraded): Adaptive Local Density [Hybrid 256-D -> LOF (k=15)]...")
    lof = LocalOutlierFactor(n_neighbors=15, metric="minkowski", novelty=True)
    lof.fit(clean_fused)
    m6_clean_scores = -lof.decision_function(clean_fused)
    m6_synth_scores = -lof.decision_function(synth_fused)
    models["Proposed Method (Adaptive LOF) [Hybrid 256-D -> LOF (k=15)]"] = (
        m6_clean_scores,
        m6_synth_scores,
    )

    # ------------------------------------------------------------
    # 3. Compute Metrics: Overall ROC-AUC, Overall AP, Slice ROC-AUC
    # ------------------------------------------------------------
    print("\n[Step 3/5] Computing overall and slice-level performance metrics...")

    slices = ["blur", "dropout", "thickening", "thinning"]
    results_records = []

    for name, (clean_sc, synth_sc) in models.items():
        # Overall binary target
        y_overall = np.concatenate([np.zeros(len(clean_sc)), np.ones(len(synth_sc))])
        scores_overall = np.concatenate([clean_sc, synth_sc])

        roc_overall = roc_auc_score(y_overall, scores_overall)
        ap_overall = average_precision_score(y_overall, scores_overall)

        rec = {
            "Method": name,
            "overall_roc_auc": roc_overall,
            "overall_ap": ap_overall,
        }

        # Slice-level ROC-AUC
        for sl in slices:
            mask = (synth_types == sl)
            y_slice = np.concatenate([np.zeros(len(clean_sc)), np.ones(mask.sum())])
            scores_slice = np.concatenate([clean_sc, synth_sc[mask]])
            rec[sl] = roc_auc_score(y_slice, scores_slice)

        results_records.append(rec)

    results_df = pd.DataFrame(results_records)

    # ------------------------------------------------------------
    # 4. Save Overlaid ROC Curves Plot
    # ------------------------------------------------------------
    print("\n[Step 4/5] Generating overlaid ROC curve visualization...")

    ROC_PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8), dpi=300)

    # Distinct, publication-grade styling for each model
    styles = [
        {"color": "#E03131", "linestyle": "--", "linewidth": 2.0},  # Baseline 1 (Asselborn: Red dashed)
        {"color": "#1971C2", "linestyle": "-",  "linewidth": 2.4},  # Baseline 2 (Devi & Kavya: Blue solid)
        {"color": "#F08C00", "linestyle": "-.", "linewidth": 2.0},  # Baseline 3 (Agarwal: Orange dash-dot)
        {"color": "#9C36B5", "linestyle": ":",  "linewidth": 2.2},  # Baseline 4 (Alevizos: Purple dotted)
        {"color": "#0CA678", "linestyle": "--", "linewidth": 2.0},  # Proposed Ablation (Teal dashed)
        {"color": "#2F9E44", "linestyle": "-",  "linewidth": 2.6},  # Proposed Upgraded LOF (Green solid)
    ]

    for (name, (clean_sc, synth_sc)), style in zip(models.items(), styles):
        y_true = np.concatenate([np.zeros(len(clean_sc)), np.ones(len(synth_sc))])
        scores = np.concatenate([clean_sc, synth_sc])
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_score = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.4f})",
            **style,
        )

    # Diagonal chance line
    plt.plot(
        [0, 1],
        [0, 1],
        color="#868E96",
        linestyle=":",
        linewidth=1.5,
        label="Chance Baseline (AUC = 0.5000)",
    )

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.04])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("Comparative ROC Curves: Competing Literature Baselines vs Proposed Pipeline", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, facecolor="#FFFFFF", edgecolor="#CED4DA", fontsize=9.2)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()

    plt.savefig(ROC_PLOT_FILE, dpi=300)
    plt.close()
    print(f"  [OK] Saved overlaid ROC curve plot to:\n       {ROC_PLOT_FILE}")

    # ------------------------------------------------------------
    # 5. Export CSV and Print Markdown Table
    # ------------------------------------------------------------
    print("\n[Step 5/5] Exporting CSV and Markdown table...")

    # Save to CSV
    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"  [OK] Saved results CSV to:\n       {OUTPUT_CSV_FILE}\n")

    # Format Markdown Table
    md_header = "| Model / Paradigm | Overall ROC-AUC | Overall AP | Blur | Dropout | Thickening | Thinning |"
    md_divider = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
    md_rows = []
    for _, r in results_df.iterrows():
        row_str = (
            f"| {r['Method']} "
            f"| **{r['overall_roc_auc']:.4f}** "
            f"| **{r['overall_ap']:.4f}** "
            f"| {r['blur']:.4f} "
            f"| {r['dropout']:.4f} "
            f"| {r['thickening']:.4f} "
            f"| {r['thinning']:.4f} |"
        )
        md_rows.append(row_str)

    markdown_table = "\n".join([md_header, md_divider] + md_rows)

    print("=" * 110)
    print("BENCHMARK COMPARATIVE RESULTS (ROC-AUC & AVERAGE PRECISION)")
    print("=" * 110)
    print(markdown_table)
    print("=" * 110)

    return results_df, markdown_table


if __name__ == "__main__":
    run_benchmark()
