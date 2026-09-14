import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import KFold


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "../data/synthetic_form_anomalies/paired_embeddings.npz"

ROC_OUTPUT = "../data/synthetic_form_anomalies/roc_curves_actual.png"
DIST_OUTPUT = "../data/synthetic_form_anomalies/clean_vs_synthetic_distribution.png"


# ============================================================
# 1. LOAD REAL EMBEDDINGS
# ============================================================

data = np.load(INPUT_FILE)

clean_emb = data["clean_embeddings"]
syn_emb = data["synthetic_embeddings"]

# Support either possible column name
if "types" in data:
    syn_types = data["types"]
elif "synthetic_types" in data:
    syn_types = data["synthetic_types"]
else:
    raise KeyError("Could not find anomaly type information in NPZ file.")


print("Clean embeddings:", clean_emb.shape)
print("Synthetic embeddings:", syn_emb.shape)


# ============================================================
# 2. 5-FOLD CROSS-VALIDATED 1-NN CLEAN SCORES
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

clean_scores = np.zeros(len(clean_emb))

for train_idx, val_idx in kf.split(clean_emb):

    nn_model = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn_model.fit(clean_emb[train_idx])

    distances, _ = nn_model.kneighbors(
        clean_emb[val_idx]
    )

    clean_scores[val_idx] = distances.ravel()


# ============================================================
# 3. SYNTHETIC ANOMALY SCORES
# ============================================================

nn_full = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn_full.fit(clean_emb)

syn_distances, _ = nn_full.kneighbors(syn_emb)

syn_scores = syn_distances.ravel()


# ============================================================
# 4. GLOBAL ROC-AUC + AP
# ============================================================

y_true = np.concatenate([
    np.zeros(len(clean_scores)),
    np.ones(len(syn_scores))
])

all_scores = np.concatenate([
    clean_scores,
    syn_scores
])

fpr, tpr, _ = roc_curve(
    y_true,
    all_scores
)

overall_auc = auc(fpr, tpr)

ap_score = average_precision_score(
    y_true,
    all_scores
)


print()
print("========================================")
print("OVERALL RESULTS")
print("========================================")
print(f"ROC-AUC: {overall_auc:.4f}")
print(f"Average Precision: {ap_score:.4f}")


# ============================================================
# 5. PER-ANOMALY ROC-AUC
# ============================================================

print()
print("========================================")
print("PER-ANOMALY RESULTS")
print("========================================")

type_results = {}

for anomaly_type in np.unique(syn_types):

    mask = syn_types == anomaly_type

    type_scores = syn_scores[mask]

    y_type = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(type_scores))
    ])

    scores_type = np.concatenate([
        clean_scores,
        type_scores
    ])

    fpr_type, tpr_type, _ = roc_curve(
        y_type,
        scores_type
    )

    auc_type = auc(
        fpr_type,
        tpr_type
    )

    type_results[str(anomaly_type)] = auc_type

    print(
        f"{str(anomaly_type).capitalize():12s} "
        f"AUC = {auc_type:.4f} "
        f"(n={len(type_scores)})"
    )


# ============================================================
# FIGURE 1
# ROC CURVES
# ============================================================

plt.figure(figsize=(8, 6))

# Overall curve
plt.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f"Overall 1-NN (AUC = {overall_auc:.4f})"
)


# Individual anomaly curves
for anomaly_type in np.unique(syn_types):

    mask = syn_types == anomaly_type

    type_scores = syn_scores[mask]

    y_type = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(type_scores))
    ])

    scores_type = np.concatenate([
        clean_scores,
        type_scores
    ])

    fpr_type, tpr_type, _ = roc_curve(
        y_type,
        scores_type
    )

    auc_type = auc(
        fpr_type,
        tpr_type
    )

    plt.plot(
        fpr_type,
        tpr_type,
        linewidth=1.8,
        label=f"{str(anomaly_type).capitalize()} "
              f"(AUC = {auc_type:.4f})"
    )


# Random baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random chance (AUC = 0.5000)"
)


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves: Synthetic Anomaly Benchmark"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    ROC_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    clean_scores,
    bins=35,
    density=True,
    alpha=0.45,
    label="Clean IAM Forms"
)

plt.hist(
    syn_scores,
    bins=35,
    density=True,
    alpha=0.45,
    label="Synthetic Anomalous Forms"
)

plt.xlabel(
    "1-Nearest Neighbor Euclidean Distance"
)

plt.ylabel("Density")

plt.title(
    "Anomaly Score Distribution: Clean vs Synthetic"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    DIST_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. SUMMARY
# ============================================================

print()
print("========================================")
print("FILES CREATED")
print("========================================")

print(ROC_OUTPUT)
print(DIST_OUTPUT)

print()
print("Done.")