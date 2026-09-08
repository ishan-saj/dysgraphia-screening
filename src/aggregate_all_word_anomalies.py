from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORD_ANOMALY_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "all_samples_word_anomaly.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "all_samples_word_summary.csv"
)


# ============================================================
# AGGREGATE ONE SAMPLE
# ============================================================

def aggregate_sample(group):

    scores = pd.to_numeric(
        group["word_anomaly_score"],
        errors="coerce"
    ).dropna()

    if len(scores) == 0:
        return None

    # Top words
    top_words_df = (
        group
        .sort_values(
            "word_anomaly_score",
            ascending=False
        )
        .head(10)
    )

    top_words = []

    for _, row in top_words_df.iterrows():

        word = str(row["transcription"])
        score = float(row["word_anomaly_score"])

        top_words.append(
            f"{word} ({score:.3f})"
        )

    # Top feature
    feature_counts = (
        group["main_anomaly_feature"]
        .dropna()
        .value_counts()
    )

    if len(feature_counts) > 0:

        most_common_feature = (
            feature_counts.index[0]
        )

        most_common_feature_count = int(
            feature_counts.iloc[0]
        )

    else:

        most_common_feature = ""
        most_common_feature_count = 0

    # Top 5 / Top 10
    top_5 = scores.nlargest(
        min(5, len(scores))
    )

    top_10 = scores.nlargest(
        min(10, len(scores))
    )

    return {
        "sample_id": str(
            group["form_id"].iloc[0]
        ),

        "total_handwriting_words": int(
            len(scores)
        ),

        "mean_word_anomaly": float(
            scores.mean()
        ),

        "median_word_anomaly": float(
            scores.median()
        ),

        "std_word_anomaly": float(
            scores.std()
        ) if len(scores) > 1 else 0.0,

        "minimum_word_anomaly": float(
            scores.min()
        ),

        "maximum_word_anomaly": float(
            scores.max()
        ),

        "top_5_mean_anomaly": float(
            top_5.mean()
        ),

        "top_10_mean_anomaly": float(
            top_10.mean()
        ),

        "75th_percentile": float(
            np.percentile(scores, 75)
        ),

        "90th_percentile": float(
            np.percentile(scores, 90)
        ),

        "95th_percentile": float(
            np.percentile(scores, 95)
        ),

        "words_score_ge_1": int(
            (scores >= 1.0).sum()
        ),

        "words_score_ge_1_25": int(
            (scores >= 1.25).sum()
        ),

        "words_score_ge_1_5": int(
            (scores >= 1.5).sum()
        ),

        "most_common_anomaly_feature":
            most_common_feature,

        "most_common_feature_count":
            most_common_feature_count,

        "top_10_words":
            "; ".join(top_words),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ALL-SAMPLE WORD ANOMALY AGGREGATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not WORD_ANOMALY_FILE.exists():

        print("\nERROR:")
        print("Input file not found:")
        print(WORD_ANOMALY_FILE)

        return

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading word anomaly results...")

    df = pd.read_csv(
        WORD_ANOMALY_FILE
    )

    print(
        f"Word-level rows: {len(df)}"
    )

    print(
        f"Input samples/forms: "
        f"{df['form_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    required_columns = [
        "form_id",
        "transcription",
        "word_anomaly_score",
        "main_anomaly_feature",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print("\nERROR: Missing columns:")
        print(missing)

        return

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    print("\nAggregating by sample...")

    summaries = []

    grouped = df.groupby(
        "form_id",
        sort=False
    )

    total_samples = len(grouped)

    for index, (_, group) in enumerate(
        grouped,
        start=1
    ):

        summary = aggregate_sample(
            group
        )

        if summary is not None:

            summaries.append(
                summary
            )

        if index % 100 == 0:

            print(
                f"Processed samples: "
                f"{index}/{total_samples}"
            )

    result = pd.DataFrame(
        summaries
    )

    # --------------------------------------------------------
    # Sort by word anomaly
    # --------------------------------------------------------

    result = result.sort_values(
        "mean_word_anomaly",
        ascending=False
    ).reset_index(
        drop=True
    )

    # Add rank
    result.insert(
        0,
        "word_anomaly_rank",
        np.arange(
            1,
            len(result) + 1
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("AGGREGATION COMPLETED")
    print("=" * 60)

    print(
        f"\nSamples aggregated: "
        f"{len(result)}"
    )

    print(
        f"Words aggregated: "
        f"{len(df)}"
    )

    print(
        f"\nOutput file:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Top 20 samples
    # --------------------------------------------------------

    print("\nTOP 20 SAMPLES BY MEAN WORD ANOMALY")
    print("-" * 60)

    columns = [
        "word_anomaly_rank",
        "sample_id",
        "total_handwriting_words",
        "mean_word_anomaly",
        "maximum_word_anomaly",
        "top_5_mean_anomaly",
        "words_score_ge_1",
        "most_common_anomaly_feature",
    ]

    print(
        result[columns]
        .head(20)
        .to_string(index=False)
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()