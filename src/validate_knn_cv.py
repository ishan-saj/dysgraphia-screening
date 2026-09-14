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
    / "knn_cv_results.csv"
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

print(
    "Clean embeddings:",
    clean_embeddings.shape
)

print(
    "Synthetic embeddings:",
    synthetic_embeddings.shape
)


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

k_values = [
    1,
    3,
    5,
    10,
    20,
    50
]

all_results = []


for k in k_values:

    print("\n" + "=" * 60)
    print(f"Testing k = {k}")
    print("=" * 60)

    clean_scores = np.zeros(
        len(clean_embeddings)
    )

    # --------------------------------------------------------
    # OUT-OF-FOLD CLEAN SCORES
    # --------------------------------------------------------

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

        # Need at least k neighbors
        knn = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean"
        )

        knn.fit(
            train_embeddings
        )

        distances, _ = knn.kneighbors(
            val_embeddings
        )

        # Mean distance to k nearest NORMAL samples
        scores = distances.mean(
            axis=1
        )

        clean_scores[val_idx] = scores

        print(
            f"Fold {fold}/5 completed"
        )


    # --------------------------------------------------------
    # SYNTHETIC SCORES
    # --------------------------------------------------------

    # Train on ALL clean samples
    knn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean"
    )

    knn.fit(
        clean_embeddings
    )

    synthetic_distances, _ = knn.kneighbors(
        synthetic_embeddings
    )

    synthetic_scores = (
        synthetic_distances.mean(
            axis=1
        )
    )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    y_true = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(synthetic_scores))
    ])

    scores = np.concatenate([
        clean_scores,
        synthetic_scores
    ])


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_true,
        scores
    )

    ap = average_precision_score(
        y_true,
        scores
    )


    print(
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"Average Precision: {ap:.4f}"
    )


    all_results.append({
        "k": k,
        "roc_auc": auc,
        "average_precision": ap
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 60)
print("5-FOLD CROSS-VALIDATED k-NN RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST K
# ============================================================

best = results_df.iloc[0]

print("\n")
print("=" * 60)
print("BEST VALIDATED k")
print("=" * 60)

print(
    f"k = {int(best['k'])}"
)

print(
    f"ROC-AUC = {best['roc_auc']:.4f}"
)

print(
    f"Average Precision = "
    f"{best['average_precision']:.4f}"
)

print("\nSaved to:")

print(
    OUTPUT_FILE
)