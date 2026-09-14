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
# LOAD EMBEDDINGS
# ============================================================

print("Loading embeddings...")

data = np.load(
    EMBEDDING_FILE,
    allow_pickle=True
)

clean_embeddings = data[
    "clean_embeddings"
]

synthetic_embeddings = data[
    "synthetic_embeddings"
]

synthetic_types = data[
    "synthetic_types"
]

synthetic_types = np.array(
    [str(x) for x in synthetic_types]
)

print(
    "Clean:",
    clean_embeddings.shape
)

print(
    "Synthetic:",
    synthetic_embeddings.shape
)


# ============================================================
# SETTINGS
# ============================================================

K = 1

N_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# OUT-OF-FOLD CLEAN SCORES
# ============================================================

print("\nCalculating out-of-fold clean scores...")

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

clean_scores = np.zeros(
    len(clean_embeddings)
)


for fold, (train_idx, val_idx) in enumerate(
    kf.split(clean_embeddings),
    start=1
):

    train_embeddings = (
        clean_embeddings[train_idx]
    )

    val_embeddings = (
        clean_embeddings[val_idx]
    )

    knn = NearestNeighbors(
        n_neighbors=K,
        metric="euclidean"
    )

    knn.fit(
        train_embeddings
    )

    distances, _ = knn.kneighbors(
        val_embeddings
    )

    clean_scores[val_idx] = (
        distances.mean(axis=1)
    )

    print(
        f"Fold {fold}/{N_SPLITS} completed"
    )


# ============================================================
# FIT ON ALL CLEAN DATA
# ============================================================

print("\nCalculating synthetic anomaly scores...")

knn = NearestNeighbors(
    n_neighbors=K,
    metric="euclidean"
)

knn.fit(
    clean_embeddings
)

synthetic_distances, _ = knn.kneighbors(
    synthetic_embeddings
)

synthetic_scores = (
    synthetic_distances.mean(axis=1)
)


# ============================================================
# EVALUATE EACH TYPE
# ============================================================

anomaly_types = sorted(
    np.unique(synthetic_types)
)

results = []


for anomaly_type in anomaly_types:

    print("\n" + "=" * 60)

    print(
        f"Evaluating: {anomaly_type}"
    )

    mask = (
        synthetic_types
        == anomaly_type
    )

    type_scores = (
        synthetic_scores[mask]
    )

    type_labels = np.ones(
        len(type_scores)
    )

    clean_labels = np.zeros(
        len(clean_scores)
    )

    y_true = np.concatenate([
        clean_labels,
        type_labels
    ])

    scores = np.concatenate([
        clean_scores,
        type_scores
    ])

    auc = roc_auc_score(
        y_true,
        scores
    )

    ap = average_precision_score(
        y_true,
        scores
    )

    results.append({
        "anomaly_type": anomaly_type,
        "count": len(type_scores),
        "mean_clean_score": clean_scores.mean(),
        "mean_anomaly_score": type_scores.mean(),
        "roc_auc": auc,
        "average_precision": ap
    })

    print(
        f"Samples: {len(type_scores)}"
    )

    print(
        f"Mean clean score: "
        f"{clean_scores.mean():.4f}"
    )

    print(
        f"Mean anomaly score: "
        f"{type_scores.mean():.4f}"
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"Average Precision: {ap:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 70)
print("PER-ANOMALY-TYPE RESULTS")
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