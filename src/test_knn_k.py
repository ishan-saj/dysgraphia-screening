from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from sklearn.neighbors import NearestNeighbors


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
    / "knn_k_comparison.csv"
)


# ============================================================
# K VALUES
# ============================================================

K_VALUES = [
    1,
    3,
    5,
    10,
    20,
    50
]


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 65)
print("LOADING PAIRED EMBEDDINGS")
print("=" * 65)

if not EMBEDDING_FILE.exists():

    print(
        "\nEmbedding file not found:"
    )

    print(
        EMBEDDING_FILE
    )

    raise SystemExit


data = np.load(
    EMBEDDING_FILE,
    allow_pickle=True
)

clean_embeddings = (
    data["clean_embeddings"]
    .astype(np.float32)
)

synthetic_embeddings = (
    data["synthetic_embeddings"]
    .astype(np.float32)
)

synthetic_types = (
    data["synthetic_types"]
)


print(
    "Clean embeddings:",
    clean_embeddings.shape
)

print(
    "Synthetic embeddings:",
    synthetic_embeddings.shape
)


# ============================================================
# TEST K VALUES
# ============================================================

results = []


for k in K_VALUES:

    print(
        f"\nTesting k = {k}"
    )

    # --------------------------------------------------------
    # Fit k-NN on ALL clean embeddings
    # --------------------------------------------------------

    knn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean"
    )

    knn.fit(
        clean_embeddings
    )


    # ========================================================
    # SYNTHETIC SCORES
    # ========================================================

    synthetic_distances, _ = (
        knn.kneighbors(
            synthetic_embeddings
        )
    )

    # For synthetic samples there is no
    # self-neighbor, so take the kth neighbor.
    synthetic_scores = (
        synthetic_distances[:, -1]
    )


    # ========================================================
    # CLEAN SCORES
    # ========================================================

    clean_distances, _ = (
        knn.kneighbors(
            clean_embeddings
        )
    )

    # Because every clean sample is part of
    # the reference set, the first neighbor is
    # the sample itself with distance = 0.
    #
    # Therefore remove the self-neighbor and
    # use the kth actual neighboring clean sample.

    clean_scores = (
        clean_distances[:, k]
    )


    # ========================================================
    # COMBINE
    # ========================================================

    y_true = np.concatenate([

        np.zeros(
            len(clean_scores)
        ),

        np.ones(
            len(synthetic_scores)
        )
    ])


    scores = np.concatenate([

        clean_scores,

        synthetic_scores
    ])


    # ========================================================
    # METRICS
    # ========================================================

    auc = roc_auc_score(
        y_true,
        scores
    )

    ap = average_precision_score(
        y_true,
        scores
    )


    results.append({

        "k":
            k,

        "roc_auc":
            auc,

        "average_precision":
            ap,

        "mean_clean_score":
            clean_scores.mean(),

        "mean_synthetic_score":
            synthetic_scores.mean()
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


# Keep k in numerical order for the CSV
results_df = results_df.sort_values(
    "k"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")
print("=" * 65)
print("k-NN K-VALUE COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST K
# ============================================================

best = results_df.loc[
    results_df["roc_auc"].idxmax()
]


print("\n")
print("=" * 65)
print("BEST k")
print("=" * 65)

print(
    f"k = {int(best['k'])}"
)

print(
    f"ROC-AUC = "
    f"{best['roc_auc']:.4f}"
)

print(
    f"Average Precision = "
    f"{best['average_precision']:.4f}"
)


# ============================================================
# OUTPUT
# ============================================================

print("\nSaved to:")

print(
    OUTPUT_FILE
)

print("\nDone.")