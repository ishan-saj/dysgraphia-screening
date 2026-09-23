from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOF_FILE = (
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

    if not LOF_FILE.exists():
        print("\nERROR: anomaly_scores.csv not found.")
        print(LOF_FILE)
        return

    if not WORD_FILE.exists():
        print("\nERROR: all_samples_word_summary.csv not found.")
        print(WORD_FILE)
        return

    # --------------------------------------------------------
    # Load LOF form-level scores
    # --------------------------------------------------------

    print("\nLoading form-level LOF anomaly scores...")

    lof = pd.read_csv(LOF_FILE)

    print(
        f"LOF samples: "
        f"{len(lof)}"
    )

    # --------------------------------------------------------
    # Load word-level summaries
    # --------------------------------------------------------

    print("\nLoading word-level sample summaries...")

    word = pd.read_csv(WORD_FILE)

    print(
        f"Word-anomaly samples: "
        f"{len(word)}"
    )

    # --------------------------------------------------------
    # Display columns
    # --------------------------------------------------------

    print("\nLOF columns:")
    print(list(lof.columns))

    print("\nWord columns:")
    print(list(word.columns))

    # --------------------------------------------------------
    # Detect LOF score columns
    # --------------------------------------------------------

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
            if column in lof.columns
        ),
        None
    )

    normalized_column = next(
        (
            column
            for column in possible_normalized
            if column in lof.columns
        ),
        None
    )

    if raw_column is None:
        print(
            "\nERROR: Could not find raw LOF "
            "anomaly score column."
        )
        return

    if normalized_column is None:
        print(
            "\nERROR: Could not find normalized LOF "
            "anomaly score column."
        )
        return

    # --------------------------------------------------------
    # Rename for clarity
    # --------------------------------------------------------

    lof = lof.rename(
        columns={
            raw_column:
                "lof_raw_anomaly",

            normalized_column:
                "lof_anomaly_score",
        }
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print("\nMerging datasets...")

    merged = pd.merge(
        lof,
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

    check_columns = [
        "lof_anomaly_score",
        "mean_word_anomaly",
        "maximum_word_anomaly",
        "top_5_mean_anomaly",
    ]

    available_check_columns = [
        column
        for column in check_columns
        if column in merged.columns
    ]

    print(
        merged[
            available_check_columns
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
    print("CORRELATION WITH LOF ANOMALY")
    print("=" * 65)

    correlations = []

    for column in correlation_columns:

        if column not in merged.columns:
            continue

        x = pd.to_numeric(
            merged["lof_anomaly_score"],
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

    merged["lof_rank_percentile"] = (
        merged["lof_anomaly_score"]
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
    #
    # 50% form-level LOF evidence
    # 50% word-level evidence
    #
    # This is a relative evidence-integration score.
    # It is NOT a probability or diagnosis.
    # --------------------------------------------------------

    merged["combined_anomaly_score"] = (
        0.5 * merged["lof_rank_percentile"]
        +
        0.5 * merged["word_rank_percentile"]
    )

    # --------------------------------------------------------
    # Agreement category
    # --------------------------------------------------------

    def classify(row):

        lof_high = (
            row["lof_rank_percentile"] >= 75
        )

        word_high = (
            row["word_rank_percentile"] >= 75
        )

        if lof_high and word_high:
            return "High in both"

        if lof_high and not word_high:
            return "High LOF only"

        if word_high and not lof_high:
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
        "lof_anomaly_score",
        "mean_word_anomaly",
        "maximum_word_anomaly",
        "top_5_mean_anomaly",
        "combined_anomaly_score",
        "anomaly_agreement",
    ]

    available_display_columns = [
        column
        for column in display_columns
        if column in merged.columns
    ]

    print(
        merged[
            available_display_columns
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
    # Top LOF samples
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("TOP 10 LOF ANOMALY SAMPLES")
    print("=" * 65)

    print(
        merged
        .sort_values(
            "lof_anomaly_score",
            ascending=False
        )[
            [
                "sample_id",
                "lof_anomaly_score",
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
                "lof_anomaly_score",
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
        f"\nFinal merged samples: "
        f"{len(merged)}"
    )

    print(
        f"Output:\n"
        f"{OUTPUT_FILE}"
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()
