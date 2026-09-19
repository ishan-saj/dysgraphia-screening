from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


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
# SETTINGS
# ============================================================

N_FOLDS = 5

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

source_indices = (
    data["source_indices"]
    .astype(int)
)


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
    len(synthetic_embeddings)
)

print(
    "Unique synthetic source forms:",
    len(np.unique(source_indices))
)


# ============================================================
# VERIFY
# ============================================================

assert len(synthetic_embeddings) == len(
    synthetic_types
)

assert len(synthetic_embeddings) == len(
    source_indices
)

assert np.all(
    source_indices >= 0
)

assert np.all(
    source_indices < len(clean_embeddings)
)

print(
    "\nEmbedding data verified."
)


# ============================================================
# K-FOLD
# ============================================================

kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=42
)


# ============================================================
# STORAGE
# ============================================================

all_results = []


# ============================================================
# TEST EACH K
# ============================================================

for k in K_VALUES:

    print("\n")
    print("=" * 65)
    print(f"TESTING k = {k}")
    print("=" * 65)


    fold_clean_scores = []

    fold_synthetic_scores = []


    # ========================================================
    # 5 FOLDS
    # ========================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        kf.split(clean_embeddings),
        start=1
    ):

        train_embeddings = (
            clean_embeddings[
                train_idx
            ]
        )

        test_clean_embeddings = (
            clean_embeddings[
                test_idx
            ]
        )


        # ----------------------------------------------------
        # Synthetic samples whose source form
        # belongs to the held-out fold
        # ----------------------------------------------------

        synthetic_mask = np.isin(
            source_indices,
            test_idx
        )

        synthetic_idx = np.where(
            synthetic_mask
        )[0]

        test_synthetic_embeddings = (
            synthetic_embeddings[
                synthetic_idx
            ]
        )


        print(
            f"Fold {fold}/{N_FOLDS}: "
            f"train clean = {len(train_embeddings)}, "
            f"test clean = {len(test_clean_embeddings)}, "
            f"test synthetic = "
            f"{len(test_synthetic_embeddings)}"
        )


        # ----------------------------------------------------
        # k-NN
        # ----------------------------------------------------

        knn = NearestNeighbors(
            n_neighbors=k,
            metric="euclidean"
        )

        knn.fit(
            train_embeddings
        )


        # ----------------------------------------------------
        # Clean validation scores
        # ----------------------------------------------------

        clean_distances, _ = (
            knn.kneighbors(
                test_clean_embeddings
            )
        )

        clean_scores = (
            clean_distances[:, -1]
        )


        # ----------------------------------------------------
        # Synthetic validation scores
        # ----------------------------------------------------

        synthetic_distances, _ = (
            knn.kneighbors(
                test_synthetic_embeddings
            )
        )

        synthetic_scores = (
            synthetic_distances[:, -1]
        )


        fold_clean_scores.extend(
            clean_scores
        )

        fold_synthetic_scores.extend(
            synthetic_scores
        )


    # ========================================================
    # COMBINE ALL HELD-OUT RESULTS
    # ========================================================

    clean_scores = np.asarray(
        fold_clean_scores
    )

    synthetic_scores = np.asarray(
        fold_synthetic_scores
    )


    y_true = np.concatenate([

        np.zeros(
            len(clean_scores),
            dtype=int
        ),

        np.ones(
            len(synthetic_scores),
            dtype=int
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


    print(
        f"\nROC-AUC: {auc:.4f}"
    )

    print(
        f"Average Precision: {ap:.4f}"
    )


    all_results.append({

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
    all_results
)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False
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
print("5-FOLD CROSS-VALIDATED k-NN RESULTS")
print("=" * 65)

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
print("=" * 65)
print("BEST VALIDATED k")
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