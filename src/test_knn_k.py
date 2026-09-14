from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "anomaly_score_comparison.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "knn_k_comparison.csv"
)


# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# The previous CSV contains k-NN score only for k=5.
# Therefore we need the actual embeddings to test different k.
#
# This script reconstructs them by using the existing
# clean/synthetic score CSV only as a source of sample IDs,
# so instead we will use the previously generated paired
# results if embedding data is available.
#
# Check for saved embedding file first.
# ------------------------------------------------------------

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "paired_embeddings.npz"
)


if not EMBEDDING_FILE.exists():

    print("=" * 60)
    print("Embedding file not found.")
    print("=" * 60)

    print(
        "\nWe need the clean and synthetic embeddings "
        "to test different k values."
    )

    print(
        "\nNext we will generate this file from the "
        "existing model without retraining."
    )

    print(
        "\nMissing:"
    )

    print(
        EMBEDDING_FILE
    )

    raise SystemExit


# ------------------------------------------------------------
# LOAD EMBEDDINGS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# TEST DIFFERENT K VALUES
# ------------------------------------------------------------

k_values = [
    1,
    3,
    5,
    10,
    20,
    50
]

results = []


for k in k_values:

    print(
        f"\nTesting k = {k}"
    )

    knn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean"
    )

    knn.fit(
        clean_embeddings
    )

    distances, _ = knn.kneighbors(
        synthetic_embeddings
    )

    # Mean distance to k nearest clean samples
    synthetic_scores = distances.mean(
        axis=1
    )

    # Calculate equivalent scores for clean samples
    clean_distances, _ = knn.kneighbors(
        clean_embeddings
    )

    clean_scores = clean_distances.mean(
        axis=1
    )

    y_true = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(synthetic_scores))
    ])

    scores = np.concatenate([
        clean_scores,
        synthetic_scores
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
        "k": k,
        "roc_auc": auc,
        "average_precision": ap
    })


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

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

print("\n")
print("=" * 60)
print("k-NN K-VALUE COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nBest k by ROC-AUC:")

best = results_df.iloc[0]

print(
    f"k = {int(best['k'])}"
)

print(
    f"ROC-AUC = {best['roc_auc']:.4f}"
)

print(
    f"AP = {best['average_precision']:.4f}"
)

print("\nSaved to:")

print(
    OUTPUT_FILE
)