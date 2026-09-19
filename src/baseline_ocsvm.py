from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
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
    / "baseline_ocsvm_results.csv"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("Loading embeddings...")

data = np.load(EMBEDDING_FILE, allow_pickle=True)

clean_embeddings = data["clean_embeddings"]
synthetic_embeddings = data["synthetic_embeddings"]
synthetic_types = np.array([str(x) for x in data["synthetic_types"]])
source_indices = data["source_indices"].astype(int)

print("Clean:", clean_embeddings.shape)
print("Synthetic:", synthetic_embeddings.shape)
print("Source indices:", source_indices.shape)


# ============================================================
# 5-FOLD PAIRED CROSS-VALIDATION
#
# Each clean sample belongs to one fold.
#
# For each fold:
#   - Train OCSVM only on the other 4 clean folds
#   - Score clean samples in the held-out fold
#   - Score synthetic samples whose source clean sample
#     belongs to that same held-out fold
#
# Therefore the synthetic sample is evaluated against a
# reference model that has NOT seen its corresponding clean
# source sample.
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

clean_scores = np.zeros(len(clean_embeddings))
synthetic_scores = np.zeros(len(synthetic_embeddings))

# Store which fold each clean sample belongs to
fold_assignments = np.zeros(len(clean_embeddings), dtype=int)

for fold, (train_idx, val_idx) in enumerate(
    kf.split(clean_embeddings),
    start=1
):

    print(f"\nFold {fold}/5")

    # --------------------------------------------------------
    # Record validation fold
    # --------------------------------------------------------

    fold_assignments[val_idx] = fold - 1

    # --------------------------------------------------------
    # Standardize using ONLY training clean samples
    # --------------------------------------------------------

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        clean_embeddings[train_idx]
    )

    clean_val_scaled = scaler.transform(
        clean_embeddings[val_idx]
    )

    # --------------------------------------------------------
    # Train One-Class SVM on clean training samples
    # --------------------------------------------------------

    ocsvm = OneClassSVM(
        kernel="rbf",
        nu=0.1,
        gamma="scale"
    )

    ocsvm.fit(train_scaled)

    # --------------------------------------------------------
    # Clean validation scores
    #
    # decision_function:
    # positive = normal
    # negative = anomalous
    #
    # Flip sign so:
    # higher score = more anomalous
    # --------------------------------------------------------

    clean_scores[val_idx] = (
        -ocsvm.decision_function(clean_val_scaled)
    )

    # --------------------------------------------------------
    # Synthetic samples whose clean source belongs to
    # this validation fold
    # --------------------------------------------------------

    synthetic_val_mask = (
        fold_assignments[source_indices] == (fold - 1)
    )

    synthetic_val_idx = np.where(
        synthetic_val_mask
    )[0]

    if len(synthetic_val_idx) > 0:

        synthetic_val_scaled = scaler.transform(
            synthetic_embeddings[synthetic_val_idx]
        )

        synthetic_scores[synthetic_val_idx] = (
            -ocsvm.decision_function(
                synthetic_val_scaled
            )
        )

    print(
        f"Clean validation samples: {len(val_idx)}"
    )

    print(
        f"Synthetic paired samples: "
        f"{len(synthetic_val_idx)}"
    )


# ============================================================
# CHECK THAT ALL SYNTHETIC SAMPLES WERE SCORED
# ============================================================

if np.any(synthetic_scores == 0):
    print(
        "\nWARNING: Some synthetic samples have "
        "zero scores. Check source_indices."
    )


# ============================================================
# OVERALL METRICS
# ============================================================

y_true = np.concatenate([
    np.zeros(len(clean_scores)),
    np.ones(len(synthetic_scores))
])

scores = np.concatenate([
    clean_scores,
    synthetic_scores
])

overall_auc = roc_auc_score(
    y_true,
    scores
)

overall_ap = average_precision_score(
    y_true,
    scores
)


print("\n" + "=" * 60)
print("ONE-CLASS SVM — PAIRED 5-FOLD EVALUATION")
print("=" * 60)

print(
    f"Overall ROC-AUC: {overall_auc:.4f}"
)

print(
    f"Overall AP:      {overall_ap:.4f}"
)


# ============================================================
# PER-ANOMALY-TYPE METRICS
# ============================================================

results = [{
    "anomaly_type": "overall",
    "roc_auc": overall_auc,
    "average_precision": overall_ap
}]

print("\nPer anomaly type:")

for anomaly_type in sorted(
    np.unique(synthetic_types)
):

    mask = synthetic_types == anomaly_type

    type_scores = synthetic_scores[mask]

    y_type = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(type_scores))
    ])

    scores_type = np.concatenate([
        clean_scores,
        type_scores
    ])

    auc_type = roc_auc_score(
        y_type,
        scores_type
    )

    ap_type = average_precision_score(
        y_type,
        scores_type
    )

    results.append({
        "anomaly_type": anomaly_type,
        "roc_auc": auc_type,
        "average_precision": ap_type
    })

    print(
        f"{anomaly_type:12s} "
        f"ROC-AUC = {auc_type:.4f}  "
        f"AP = {ap_type:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)