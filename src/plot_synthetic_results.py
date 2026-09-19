import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import KFold


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    "../data/synthetic_form_anomalies/paired_embeddings.npz"
)

ROC_OUTPUT = (
    "../data/synthetic_form_anomalies/roc_curves_actual.png"
)

DIST_OUTPUT = (
    "../data/synthetic_form_anomalies/"
    "clean_vs_synthetic_distribution.png"
)


# ============================================================
# SETTINGS
# ============================================================

# Final proposed method:
# distance to the 5th nearest clean neighbour
K = 5

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


print(
    "Clean embeddings:",
    clean_emb.shape
)

print(
    "Synthetic embeddings:",
    syn_emb.shape
)

print(
    "Source indices:",
    source_indices.shape
)


# ============================================================
# 2. VALIDATE PAIRING
# ============================================================

if len(syn_emb) != len(source_indices):

    raise ValueError(
        "Synthetic embeddings and source indices "
        "have different lengths."
    )


if len(syn_emb) != len(syn_types):

    raise ValueError(
        "Synthetic embeddings and anomaly types "
        "have different lengths."
    )


if np.any(source_indices < 0) or np.any(
    source_indices >= len(clean_emb)
):

    raise ValueError(
        "Invalid source index detected."
    )


# ============================================================
# 3. PAIRED 5-FOLD CROSS-VALIDATED 5-NN SCORES
# ============================================================
#
# For every fold:
#
# TRAINING CLEAN
#       ↓
# 5-NN reference
#
# VALIDATION CLEAN
#       ↓
# clean anomaly scores
#
# SYNTHETIC VERSIONS OF VALIDATION CLEAN
#       ↓
# synthetic anomaly scores
#
# Anomaly score =
# distance to the 5th nearest clean neighbour.
#
# The source clean sample of a synthetic image is never
# included in the reference set for that synthetic image.
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


clean_evaluated = np.zeros(
    len(clean_emb),
    dtype=bool
)

synthetic_evaluated = np.zeros(
    len(syn_emb),
    dtype=bool
)


for fold, (train_idx, val_idx) in enumerate(
    kf.split(clean_emb),
    start=1
):

    print(
        f"Fold {fold}/{N_SPLITS}"
    )


    # --------------------------------------------------------
    # Clean training reference
    # --------------------------------------------------------

    reference_emb = clean_emb[
        train_idx
    ]


    # --------------------------------------------------------
    # Fit 5-NN model
    # --------------------------------------------------------

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


    # IMPORTANT:
    # Use distance to the K-th nearest neighbour,
    # not the mean distance.
    clean_scores[val_idx] = (
        clean_distances[:, -1]
    )

    clean_evaluated[val_idx] = True


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


    print(
        f"  Clean validation samples: "
        f"{len(val_idx)}"
    )

    print(
        f"  Synthetic paired samples: "
        f"{len(synthetic_fold_idx)}"
    )


    # --------------------------------------------------------
    # Score corresponding synthetic samples using the SAME
    # clean reference set.
    # --------------------------------------------------------

    if len(synthetic_fold_idx) > 0:

        val_syn_emb = syn_emb[
            synthetic_fold_idx
        ]

        syn_distances, _ = nn_model.kneighbors(
            val_syn_emb
        )


        # IMPORTANT:
        # Use distance to the K-th nearest neighbour.
        synthetic_scores[
            synthetic_fold_idx
        ] = syn_distances[:, -1]


        synthetic_evaluated[
            synthetic_fold_idx
        ] = True


# ============================================================
# 4. VALIDATION CHECK
# ============================================================

if not np.all(clean_evaluated):

    missing = np.sum(
        ~clean_evaluated
    )

    raise RuntimeError(
        f"{missing} clean samples were not evaluated."
    )


if not np.all(synthetic_evaluated):

    missing = np.sum(
        ~synthetic_evaluated
    )

    raise RuntimeError(
        f"{missing} synthetic samples were not evaluated."
    )


print()
print(
    "All paired scores calculated."
)


# ============================================================
# 5. OVERALL PAIRED ROC-AUC + AP
# ============================================================
#
# Each synthetic sample is compared against the anomaly
# score of its corresponding clean source sample.
#
# This is the same paired protocol used by
# evaluate_anomaly_types.py.
#
# ============================================================

paired_clean_scores = (
    clean_scores[
        source_indices
    ]
)

paired_synthetic_scores = (
    synthetic_scores
)


y_true = np.concatenate([
    np.zeros(
        len(paired_clean_scores)
    ),
    np.ones(
        len(paired_synthetic_scores)
    )
])


all_scores = np.concatenate([
    paired_clean_scores,
    paired_synthetic_scores
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
print("OVERALL PAIRED RESULTS")
print("========================================")

print(
    f"ROC-AUC: {overall_auc:.4f}"
)

print(
    f"Average Precision: {ap_score:.4f}"
)


# ============================================================
# 6. PER-ANOMALY RESULTS
# ============================================================

print()
print("========================================")
print("PER-ANOMALY RESULTS")
print("========================================")


type_results = {}


for anomaly_type in sorted(
    np.unique(syn_types)
):

    mask = (
        syn_types == anomaly_type
    )


    # Synthetic scores for this anomaly type
    type_syn_scores = (
        synthetic_scores[mask]
    )


    # Source clean indices corresponding to
    # these synthetic anomalies
    type_source_indices = (
        source_indices[mask]
    )


    # Corresponding clean scores
    type_clean_scores = (
        clean_scores[
            type_source_indices
        ]
    )


    # --------------------------------------------------------
    # Labels
    #
    # 0 = clean
    # 1 = synthetic anomaly
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

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

        "fpr":
            fpr_type,

        "tpr":
            tpr_type,

        "auc":
            auc_type
    }


    print(
        f"{str(anomaly_type).capitalize():12s} "
        f"AUC = {auc_type:.4f} "
        f"(n={len(type_syn_scores)})"
    )


# ============================================================
# 7. ROC CURVES
# ============================================================

plt.figure(
    figsize=(8, 6)
)


# ------------------------------------------------------------
# Overall ROC curve
# ------------------------------------------------------------

plt.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=(
        f"Overall 5-NN "
        f"(AUC = {overall_auc:.4f})"
    )
)


# ------------------------------------------------------------
# Individual anomaly curves
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Random baseline
# ------------------------------------------------------------

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
# 8. SCORE DISTRIBUTION
# ============================================================
#
# For the distribution plot, use the complete clean score
# distribution and synthetic score distribution.
#
# This visualizes how the anomaly scores are distributed.
#
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
    "5-NN Anomaly Score "
    "(Distance to 5th Nearest Clean Neighbour)"
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
# 9. SUMMARY
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