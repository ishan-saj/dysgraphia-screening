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
# SETTINGS
# ============================================================

K = 1
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# 1. LOAD PAIRED EMBEDDINGS
# ============================================================

data = np.load(
    INPUT_FILE,
    allow_pickle=True
)

clean_emb = data["clean_embeddings"]
syn_emb = data["synthetic_embeddings"]
syn_types = data["synthetic_types"]
source_indices = data["source_indices"]


print("Clean embeddings:", clean_emb.shape)
print("Synthetic embeddings:", syn_emb.shape)


# ============================================================
# 2. VALIDATE PAIRING
# ============================================================

if len(syn_emb) != len(source_indices):

    raise ValueError(
        "Synthetic embeddings and source indices have different lengths."
    )


if np.any(source_indices < 0) or np.any(
    source_indices >= len(clean_emb)
):

    raise ValueError(
        "Invalid source index detected."
    )


# ============================================================
# 3. PAIRED 5-FOLD CROSS-VALIDATED 1-NN SCORES
# ============================================================
#
# For every fold:
#
# TRAINING CLEAN
#       ↓
#   1-NN reference
#
# VALIDATION CLEAN
#       ↓
# clean anomaly scores
#
# SYNTHETIC VERSIONS OF VALIDATION CLEAN
#       ↓
# synthetic anomaly scores
#
# This is the same evaluation procedure used by
# evaluate_anomaly_types.py.
#
# ============================================================

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


clean_scores = np.zeros(
    len(clean_emb),
    dtype=np.float64
)

synthetic_scores = np.zeros(
    len(syn_emb),
    dtype=np.float64
)


for fold, (train_idx, val_idx) in enumerate(
    kf.split(clean_emb),
    start=1
):

    print(
        f"Fold {fold}/{N_SPLITS} completed"
    )


    # --------------------------------------------------------
    # Clean training reference
    # --------------------------------------------------------

    reference_emb = clean_emb[
        train_idx
    ]


    nn_model = NearestNeighbors(
        n_neighbors=K,
        metric="euclidean"
    )

    nn_model.fit(
        reference_emb
    )


    # --------------------------------------------------------
    # Score validation clean samples
    # --------------------------------------------------------

    val_clean_emb = clean_emb[
        val_idx
    ]

    clean_distances, _ = nn_model.kneighbors(
        val_clean_emb
    )

    clean_scores[val_idx] = (
        clean_distances.mean(axis=1)
    )


    # --------------------------------------------------------
    # Find synthetic samples whose source clean sample
    # belongs to this validation fold.
    # --------------------------------------------------------

    validation_set = set(
        val_idx.tolist()
    )

    synthetic_fold_idx = np.array(
        [
            i
            for i, source_idx
            in enumerate(source_indices)
            if int(source_idx) in validation_set
        ],
        dtype=np.int64
    )


    # --------------------------------------------------------
    # Score those synthetic samples using the SAME
    # reference set.
    # --------------------------------------------------------

    if len(synthetic_fold_idx) > 0:

        val_syn_emb = syn_emb[
            synthetic_fold_idx
        ]

        syn_distances, _ = nn_model.kneighbors(
            val_syn_emb
        )

        synthetic_scores[
            synthetic_fold_idx
        ] = syn_distances.mean(axis=1)


print("All paired scores calculated.")


# ============================================================
# 4. OVERALL ROC-AUC + AP
# ============================================================

y_true = np.concatenate([
    np.zeros(len(clean_scores)),
    np.ones(len(synthetic_scores))
])

all_scores = np.concatenate([
    clean_scores,
    synthetic_scores
])


fpr, tpr, _ = roc_curve(
    y_true,
    all_scores
)

overall_auc = auc(
    fpr,
    tpr
)

ap_score = average_precision_score(
    y_true,
    all_scores
)


print()
print("========================================")
print("OVERALL RESULTS")
print("========================================")

print(
    f"ROC-AUC: {overall_auc:.4f}"
)

print(
    f"Average Precision: {ap_score:.4f}"
)


# ============================================================
# 5. PER-ANOMALY RESULTS
# ============================================================

print()
print("========================================")
print("PER-ANOMALY RESULTS")
print("========================================")


type_results = {}


for anomaly_type in np.unique(
    syn_types
):

    mask = (
        syn_types == anomaly_type
    )

    type_syn_scores = (
        synthetic_scores[mask]
    )

    type_source_indices = (
        source_indices[mask]
    )


    # --------------------------------------------------------
    # Compare each synthetic anomaly with its corresponding
    # clean source sample.
    # --------------------------------------------------------

    type_clean_scores = (
        clean_scores[
            type_source_indices
        ]
    )


    y_type = np.concatenate([
        np.zeros(
            len(type_clean_scores)
        ),
        np.ones(
            len(type_syn_scores)
        )
    ])


    scores_type = np.concatenate([
        type_clean_scores,
        type_syn_scores
    ])


    fpr_type, tpr_type, _ = roc_curve(
        y_type,
        scores_type
    )


    auc_type = auc(
        fpr_type,
        tpr_type
    )


    type_results[
        str(anomaly_type)
    ] = {
        "fpr": fpr_type,
        "tpr": tpr_type,
        "auc": auc_type
    }


    print(
        f"{str(anomaly_type).capitalize():12s} "
        f"AUC = {auc_type:.4f} "
        f"(n={len(type_syn_scores)})"
    )


# ============================================================
# 6. ROC CURVES
# ============================================================

plt.figure(
    figsize=(8, 6)
)


# Overall curve

plt.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=(
        f"Overall 1-NN "
        f"(AUC = {overall_auc:.4f})"
    )
)


# Individual anomaly curves

for anomaly_type, result in (
    type_results.items()
):

    plt.plot(
        result["fpr"],
        result["tpr"],
        linewidth=1.8,
        label=(
            f"{anomaly_type.capitalize()} "
            f"(AUC = {result['auc']:.4f})"
        )
    )


# Random baseline

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random chance (AUC = 0.5000)"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves: Paired Synthetic Anomaly Benchmark"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    ROC_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. SCORE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.hist(
    clean_scores,
    bins=35,
    density=True,
    alpha=0.45,
    label="Clean IAM Forms"
)


plt.hist(
    synthetic_scores,
    bins=35,
    density=True,
    alpha=0.45,
    label="Synthetic Anomalous Forms"
)


plt.xlabel(
    "1-Nearest Neighbor Euclidean Distance"
)

plt.ylabel(
    "Density"
)

plt.title(
    "Anomaly Score Distribution: Clean vs Synthetic"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    DIST_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. SUMMARY
# ============================================================

print()
print("========================================")
print("FILES CREATED")
print("========================================")

print(
    ROC_OUTPUT
)

print(
    DIST_OUTPUT
)

print()
print("Done.")