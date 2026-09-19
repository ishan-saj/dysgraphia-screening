from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "paired_embeddings.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "anomaly_type_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

K = 5
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("Loading paired embeddings...")

data = np.load(
    EMBEDDING_FILE,
    allow_pickle=True
)

clean_embeddings = data["clean_embeddings"]
synthetic_embeddings = data["synthetic_embeddings"]
synthetic_types = data["synthetic_types"]
source_forms = data["source_forms"]
source_indices = data["source_indices"]


print(
    "Clean embeddings:",
    clean_embeddings.shape
)

print(
    "Synthetic embeddings:",
    synthetic_embeddings.shape
)

print(
    "Synthetic samples:",
    len(synthetic_types)
)

print(
    "Source indices:",
    source_indices.shape
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(synthetic_embeddings) != len(source_indices):

    raise ValueError(
        "Number of synthetic embeddings does not match "
        "number of source indices."
    )


if len(synthetic_embeddings) != len(synthetic_types):

    raise ValueError(
        "Number of synthetic embeddings does not match "
        "number of anomaly types."
    )


if np.any(source_indices < 0) or np.any(
    source_indices >= len(clean_embeddings)
):

    raise ValueError(
        "Some source indices are outside the clean embedding range."
    )


# ============================================================
# CHECK THAT SOURCE MAPPING IS VALID
# ============================================================

print("\nChecking source mapping...")

for i in range(min(10, len(source_indices))):

    clean_index = source_indices[i]

    print(
        f"Synthetic {i}: "
        f"type={synthetic_types[i]} "
        f"source_form={source_forms[i]} "
        f"clean_index={clean_index}"
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================
#
# Each synthetic sample is kept with the clean sample
# it came from.
#
# For each fold:
#
# TRAIN CLEAN
#      ↓
# reference set for kNN
#
# VALIDATION CLEAN
#      ↓
# clean anomaly scores
#
# SYNTHETIC VERSIONS OF VALIDATION CLEAN
#      ↓
# synthetic anomaly scores
#
# This prevents the source clean sample from being
# accidentally included in the reference set.
#
# ============================================================

print("\nRunning paired 5-fold evaluation...")

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# STORAGE
# ============================================================

clean_scores = np.zeros(
    len(clean_embeddings),
    dtype=np.float64
)

synthetic_scores = np.zeros(
    len(synthetic_embeddings),
    dtype=np.float64
)

clean_evaluated = np.zeros(
    len(clean_embeddings),
    dtype=bool
)

synthetic_evaluated = np.zeros(
    len(synthetic_embeddings),
    dtype=bool
)


# ============================================================
# FOLD LOOP
# ============================================================

for fold, (train_indices, validation_indices) in enumerate(
    kf.split(clean_embeddings),
    start=1
):

    print(
        f"\nFold {fold}/{N_SPLITS}"
    )

    print(
        "Training clean:",
        len(train_indices)
    )

    print(
        "Validation clean:",
        len(validation_indices)
    )


    # --------------------------------------------------------
    # Training clean embeddings become the reference set.
    # --------------------------------------------------------

    reference_embeddings = (
        clean_embeddings[train_indices]
    )


    # --------------------------------------------------------
    # Fit kNN reference model.
    # --------------------------------------------------------

    n_neighbors = min(
        K,
        len(reference_embeddings)
    )

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    knn.fit(
        reference_embeddings
    )


    # --------------------------------------------------------
    # Score validation clean samples.
    # --------------------------------------------------------

    validation_clean_embeddings = (
        clean_embeddings[validation_indices]
    )

    clean_distances, _ = knn.kneighbors(
        validation_clean_embeddings
    )

    # Use distance to the K-th nearest clean neighbour.
    validation_clean_scores = (
        clean_distances[:, -1]
    )


    clean_scores[validation_indices] = (
        validation_clean_scores
    )

    clean_evaluated[validation_indices] = True


    # --------------------------------------------------------
    # Find synthetic samples whose SOURCE CLEAN SAMPLE
    # belongs to this validation fold.
    # --------------------------------------------------------

    validation_set = set(
        validation_indices.tolist()
    )

    synthetic_fold_indices = []

    for synthetic_index, source_index in enumerate(
        source_indices
    ):

        if int(source_index) in validation_set:

            synthetic_fold_indices.append(
                synthetic_index
            )


    synthetic_fold_indices = np.array(
        synthetic_fold_indices,
        dtype=np.int64
    )


    print(
        "Synthetic validation samples:",
        len(synthetic_fold_indices)
    )


    # --------------------------------------------------------
    # Score corresponding synthetic samples against the
    # SAME reference set.
    # --------------------------------------------------------

    if len(synthetic_fold_indices) > 0:

        validation_synthetic_embeddings = (
            synthetic_embeddings[
                synthetic_fold_indices
            ]
        )

        synthetic_distances, _ = knn.kneighbors(
            validation_synthetic_embeddings
        )

        # Use distance to the K-th nearest clean neighbour.
        validation_synthetic_scores = (
            synthetic_distances[:, -1]
        )

        synthetic_scores[
            synthetic_fold_indices
        ] = validation_synthetic_scores

        synthetic_evaluated[
            synthetic_fold_indices
        ] = True


# ============================================================
# VALIDATION CHECK
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


print("\nAll clean samples evaluated.")

print(
    "All synthetic samples evaluated."
)


# ============================================================
# EVALUATE EACH ANOMALY TYPE
# ============================================================

print("\n")
print("=" * 70)
print("PAIRED ANOMALY TYPE RESULTS")
print("=" * 70)


results = []


anomaly_types = sorted(
    np.unique(
        synthetic_types
    )
)


for anomaly_type in anomaly_types:

    mask = (
        synthetic_types
        == anomaly_type
    )


    type_synthetic_scores = (
        synthetic_scores[mask]
    )


    # --------------------------------------------------------
    # Use the clean samples that correspond to the synthetic
    # samples of THIS anomaly type.
    #
    # This makes the comparison paired.
    # --------------------------------------------------------

    type_source_indices = (
        source_indices[mask]
    )


    type_clean_scores = (
        clean_scores[
            type_source_indices
        ]
    )


    # --------------------------------------------------------
    # Labels:
    #
    # 0 = clean
    # 1 = synthetic anomaly
    # --------------------------------------------------------

    y_true = np.concatenate(
        [
            np.zeros(
                len(type_clean_scores)
            ),
            np.ones(
                len(type_synthetic_scores)
            )
        ]
    )


    y_scores = np.concatenate(
        [
            type_clean_scores,
            type_synthetic_scores
        ]
    )


    roc_auc = roc_auc_score(
        y_true,
        y_scores
    )

    average_precision = (
        average_precision_score(
            y_true,
            y_scores
        )
    )


    clean_mean = (
        np.mean(
            type_clean_scores
        )
    )

    synthetic_mean = (
        np.mean(
            type_synthetic_scores
        )
    )

    score_change = (
        synthetic_mean
        - clean_mean
    )


    results.append(
        {
            "anomaly_type": anomaly_type,

            "n_synthetic":
                len(type_synthetic_scores),

            "n_clean":
                len(type_clean_scores),

            "clean_mean_score":
                clean_mean,

            "synthetic_mean_score":
                synthetic_mean,

            "mean_score_change":
                score_change,

            "roc_auc":
                roc_auc,

            "average_precision":
                average_precision
        }
    )


    print(
        f"\n{anomaly_type}"
    )

    print(
        "  Clean samples:",
        len(type_clean_scores)
    )

    print(
        "  Synthetic samples:",
        len(type_synthetic_scores)
    )

    print(
        f"  Clean mean score: "
        f"{clean_mean:.6f}"
    )

    print(
        f"  Synthetic mean score: "
        f"{synthetic_mean:.6f}"
    )

    print(
        f"  Mean score change: "
        f"{score_change:+.6f}"
    )

    print(
        f"  ROC-AUC: "
        f"{roc_auc:.6f}"
    )

    print(
        f"  Average Precision: "
        f"{average_precision:.6f}"
    )


# ============================================================
# OVERALL PAIRED METRICS
# ============================================================
#
# Each synthetic sample is compared with the clean sample
# from which it was generated.
#
# This preserves the paired evaluation protocol.
# ============================================================

all_clean_scores = []

all_synthetic_scores = []


for i in range(
    len(synthetic_embeddings)
):

    clean_idx = source_indices[i]

    all_clean_scores.append(
        clean_scores[clean_idx]
    )

    all_synthetic_scores.append(
        synthetic_scores[i]
    )


all_clean_scores = np.array(
    all_clean_scores
)

all_synthetic_scores = np.array(
    all_synthetic_scores
)


# ------------------------------------------------------------
# Labels
#
# 0 = clean
# 1 = synthetic anomaly
# ------------------------------------------------------------

y_true_overall = np.concatenate(
    [
        np.zeros(
            len(all_clean_scores)
        ),
        np.ones(
            len(all_synthetic_scores)
        )
    ]
)


scores_overall = np.concatenate(
    [
        all_clean_scores,
        all_synthetic_scores
    ]
)


overall_auc = roc_auc_score(
    y_true_overall,
    scores_overall
)

overall_ap = (
    average_precision_score(
        y_true_overall,
        scores_overall
    )
)


# ============================================================
# ADD OVERALL RESULT TO CSV DATA
# ============================================================

results.append(
    {
        "anomaly_type": "overall",

        "n_synthetic":
            len(all_synthetic_scores),

        "n_clean":
            len(all_clean_scores),

        "clean_mean_score":
            np.mean(all_clean_scores),

        "synthetic_mean_score":
            np.mean(all_synthetic_scores),

        "mean_score_change":
            np.mean(all_synthetic_scores)
            - np.mean(all_clean_scores),

        "roc_auc":
            overall_auc,

        "average_precision":
            overall_ap
    }
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


print("\nSaved to:")

print(
    OUTPUT_FILE
)


print("\n")
print("=" * 70)
print("OVERALL PAIRED RESULTS")
print("=" * 70)

print(
    f"Overall ROC-AUC: "
    f"{overall_auc:.6f}"
)

print(
    f"Overall AP:      "
    f"{overall_ap:.6f}"
)