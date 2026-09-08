from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "p06-052_word_anomaly.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "p06-052_sample_word_summary.csv"
)


# ============================================================
# LOAD WORD ANOMALIES
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

if df.empty:
    raise ValueError("Word anomaly CSV is empty.")


# ============================================================
# BASIC STATISTICS
# ============================================================

scores = df["word_anomaly_score"].astype(float)

sample_id = str(df["form_id"].iloc[0])

total_words = len(df)

mean_score = scores.mean()

median_score = scores.median()

std_score = scores.std()

max_score = scores.max()

min_score = scores.min()

top_5_mean = scores.nlargest(
    min(5, len(scores))
).mean()

top_10_mean = scores.nlargest(
    min(10, len(scores))
).mean()


# ============================================================
# PERCENTILES
# ============================================================

percentile_75 = np.percentile(
    scores,
    75
)

percentile_90 = np.percentile(
    scores,
    90
)

percentile_95 = np.percentile(
    scores,
    95
)


# ============================================================
# HIGH-ANOMALY WORD COUNTS
# ============================================================

high_1_0 = int(
    (scores >= 1.0).sum()
)

high_1_25 = int(
    (scores >= 1.25).sum()
)

high_1_5 = int(
    (scores >= 1.5).sum()
)


# ============================================================
# MAIN FEATURE FREQUENCY
# ============================================================

feature_counts = (
    df["main_anomaly_feature"]
    .value_counts()
)

most_common_feature = (
    feature_counts.index[0]
    if len(feature_counts) > 0
    else ""
)

most_common_feature_count = (
    int(feature_counts.iloc[0])
    if len(feature_counts) > 0
    else 0
)


# ============================================================
# TOP WORDS
# ============================================================

top_words = (
    df.sort_values(
        "word_anomaly_score",
        ascending=False
    )
    .head(10)
)

top_word_text = "; ".join(
    f"{row.transcription} ({row.word_anomaly_score:.3f})"
    for row in top_words.itertuples()
)


# ============================================================
# CREATE SUMMARY
# ============================================================

summary = pd.DataFrame([
    {

        "sample_id":
            sample_id,

        "total_handwriting_words":
            total_words,

        "mean_word_anomaly":
            round(mean_score, 6),

        "median_word_anomaly":
            round(median_score, 6),

        "std_word_anomaly":
            round(std_score, 6),

        "minimum_word_anomaly":
            round(min_score, 6),

        "maximum_word_anomaly":
            round(max_score, 6),

        "top_5_mean_anomaly":
            round(top_5_mean, 6),

        "top_10_mean_anomaly":
            round(top_10_mean, 6),

        "75th_percentile":
            round(percentile_75, 6),

        "90th_percentile":
            round(percentile_90, 6),

        "95th_percentile":
            round(percentile_95, 6),

        "words_score_ge_1":
            high_1_0,

        "words_score_ge_1_25":
            high_1_25,

        "words_score_ge_1_5":
            high_1_5,

        "most_common_anomaly_feature":
            most_common_feature,

        "most_common_feature_count":
            most_common_feature_count,

        "top_10_words":
            top_word_text,

    }
])


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print("=" * 60)
print("SAMPLE-LEVEL WORD ANOMALY AGGREGATION")
print("=" * 60)

print()
print(f"Sample: {sample_id}")
print(f"Handwriting words: {total_words}")

print()
print("Word anomaly statistics")
print("-" * 40)

print(
    f"Mean:       {mean_score:.4f}"
)

print(
    f"Median:     {median_score:.4f}"
)

print(
    f"Std:        {std_score:.4f}"
)

print(
    f"Maximum:    {max_score:.4f}"
)

print(
    f"Top-5 mean: {top_5_mean:.4f}"
)

print(
    f"Top-10 mean:{top_10_mean:.4f}"
)

print()
print("Percentiles")
print("-" * 40)

print(
    f"75th: {percentile_75:.4f}"
)

print(
    f"90th: {percentile_90:.4f}"
)

print(
    f"95th: {percentile_95:.4f}"
)

print()
print("Higher-anomaly word counts")
print("-" * 40)

print(
    f"Score >= 1.00 : {high_1_0}"
)

print(
    f"Score >= 1.25 : {high_1_25}"
)

print(
    f"Score >= 1.50 : {high_1_5}"
)

print()
print("Most common anomaly feature:")
print(
    f"{most_common_feature} "
    f"({most_common_feature_count} words)"
)

print()
print("Top anomalous words")
print("-" * 40)

for i, row in enumerate(
    top_words.itertuples(),
    start=1
):

    print(
        f"{i}. {row.transcription} "
        f"({row.word_anomaly_score:.4f})"
    )

print()
print("=" * 60)
print("AGGREGATION COMPLETED")
print("=" * 60)

print()
print("Saved to:")
print(OUTPUT_CSV)