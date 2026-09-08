from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESNET_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "anomaly_scores.csv"
)

WORD_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "all_samples_word_summary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "master_anomaly_analysis.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SAMPLE ANOMALY COMPARISON")
    print("=" * 65)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not RESNET_FILE.exists():
        print("\nERROR: anomaly_scores.csv not found.")
        print(RESNET_FILE)
        return

    if not WORD_FILE.exists():
        print("\nERROR: all_samples_word_summary.csv not found.")
        print(WORD_FILE)
        return

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading sample-level ResNet anomaly scores...")

    resnet = pd.read_csv(RESNET_FILE)

    print(
        f"ResNet samples: "
        f"{len(resnet)}"
    )

    print("\nLoading word-level sample summaries...")

    word = pd.read_csv(WORD_FILE)

    print(
        f"Word-anomaly samples: "
        f"{len(word)}"
    )

    # --------------------------------------------------------
    # Detect ResNet score columns
    # --------------------------------------------------------

    print("\nResNet columns:")
    print(list(resnet.columns))

    print("\nWord columns:")
    print(list(word.columns))

    # Existing evaluate.py normally creates:
    #
    # sample_id
    # raw_anomaly_score
    # normalized_anomaly_score

    possible_raw = [
        "raw_anomaly_score",
        "anomaly_score",
    ]

    possible_normalized = [
        "normalized_anomaly_score",
        "anomaly_score_normalized",
        "anomaly_score_0_100",
    ]

    raw_column = next(
        (
            column
            for column in possible_raw
            if column in resnet.columns
        ),
        None
    )

    normalized_column = next(
        (
            column
            for column in possible_normalized
            if column in resnet.columns
        ),
        None
    )

    if raw_column is None:
        print(
            "\nERROR: Could not find raw ResNet "
            "anomaly score column."
        )
        return

    if normalized_column is None:
        print(
            "\nERROR: Could not find normalized ResNet "
            "anomaly score column."
        )
        return

    # --------------------------------------------------------
    # Rename for clarity
    # --------------------------------------------------------

    resnet = resnet.rename(
        columns={
            raw_column: "resnet_raw_anomaly",
            normalized_column: "resnet_anomaly_score",
        }
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\nMerging datasets...")

    merged = pd.merge(
        resnet,
        word,
        on="sample_id",
        how="inner"
    )

    print(
        f"Merged samples: "
        f"{len(merged)}"
    )

    # --------------------------------------------------------
    # Check missing data
    # --------------------------------------------------------

    print("\nMissing values:")
    print(
        merged[
            [
                "resnet_anomaly_score",
                "mean_word_anomaly",
                "maximum_word_anomaly",
                "top_5_mean_anomaly",
            ]
        ]
        .isna()
        .sum()
    )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    correlation_columns = [
        "mean_word_anomaly",
        "median_word_anomaly",
        "maximum_word_anomaly",
        "top_5_mean_anomaly",
        "top_10_mean_anomaly",
        "75th_percentile",
        "90th_percentile",
        "95th_percentile",
        "words_score_ge_1",
        "words_score_ge_1_25",
        "words_score_ge_1_5",
    ]

    print("\n")
    print("=" * 65)
    print("CORRELATION WITH RESNET ANOMALY")
    print("=" * 65)

    correlations = []

    for column in correlation_columns:

        if column not in merged.columns:
            continue

        x = pd.to_numeric(
            merged["resnet_anomaly_score"],
            errors="coerce"
        )

        y = pd.to_numeric(
            merged[column],
            errors="coerce"
        )

        valid = (
            x.notna()
            & y.notna()
        )

        if valid.sum() < 2:
            continue

        correlation = x[valid].corr(
            y[valid]
        )

        correlations.append(
            {
                "metric": column,
                "pearson_correlation": correlation,
            }
        )

        print(
            f"{column:<30}"
            f"{correlation:.4f}"
        )

    correlation_df = pd.DataFrame(
        correlations
    )

    # --------------------------------------------------------
    # Percentile ranks
    # --------------------------------------------------------

    merged["resnet_rank_percentile"] = (
        merged["resnet_anomaly_score"]
        .rank(pct=True)
        * 100
    )

    merged["word_rank_percentile"] = (
        merged["mean_word_anomaly"]
        .rank(pct=True)
        * 100
    )

    merged["top5_word_rank_percentile"] = (
        merged["top_5_mean_anomaly"]
        .rank(pct=True)
        * 100
    )

    # --------------------------------------------------------
    # Combined anomaly score
    # --------------------------------------------------------
    #
    # Both systems are converted to percentile ranks.
    # Therefore neither raw scale dominates the other.
    #
    # 50% ResNet
    # 50% word-level mean
    #

    merged["combined_anomaly_score"] = (
        0.5 * merged["resnet_rank_percentile"]
        +
        0.5 * merged["word_rank_percentile"]
    )

    # --------------------------------------------------------
    # Agreement category
    # --------------------------------------------------------

    def classify(row):

        resnet_high = (
            row["resnet_rank_percentile"] >= 75
        )

        word_high = (
            row["word_rank_percentile"] >= 75
        )

        if resnet_high and word_high:
            return "High in both"

        if resnet_high and not word_high:
            return "High ResNet only"

        if word_high and not resnet_high:
            return "High word anomaly only"

        return "Low/moderate in both"

    merged["anomaly_agreement"] = (
        merged.apply(
            classify,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    merged = merged.sort_values(
        "combined_anomaly_score",
        ascending=False
    ).reset_index(drop=True)

    merged.insert(
        0,
        "combined_rank",
        np.arange(
            1,
            len(merged) + 1
        )
    )

    # --------------------------------------------------------
    # Save master file
    # --------------------------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display top samples
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("TOP 20 COMBINED ANOMALY SAMPLES")
    print("=" * 65)

    display_columns = [
        "combined_rank",
        "sample_id",
        "resnet_anomaly_score",
        "mean_word_anomaly",
        "maximum_word_anomaly",
        "top_5_mean_anomaly",
        "combined_anomaly_score",
        "anomaly_agreement",
    ]

    print(
        merged[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Agreement summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("ANOMALY AGREEMENT")
    print("=" * 65)

    agreement_counts = (
        merged["anomaly_agreement"]
        .value_counts()
    )

    print(
        agreement_counts.to_string()
    )

    # --------------------------------------------------------
    # Top ResNet samples
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("TOP 10 RESNET ANOMALY SAMPLES")
    print("=" * 65)

    print(
        merged
        .sort_values(
            "resnet_anomaly_score",
            ascending=False
        )[
            [
                "sample_id",
                "resnet_anomaly_score",
                "mean_word_anomaly",
                "combined_anomaly_score",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top word-anomaly samples
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("TOP 10 WORD ANOMALY SAMPLES")
    print("=" * 65)

    print(
        merged
        .sort_values(
            "mean_word_anomaly",
            ascending=False
        )[
            [
                "sample_id",
                "mean_word_anomaly",
                "maximum_word_anomaly",
                "top_5_mean_anomaly",
                "resnet_anomaly_score",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("COMPARISON COMPLETED")
    print("=" * 65)

    print(
        f"\nFinal samples: "
        f"{len(merged)}"
    )

    print(
        f"Output:\n"
        f"{OUTPUT_FILE}"
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()