from pathlib import Path
import pandas as pd


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
    / "p06-052_word_explanation.csv"
)

TOP_K = 10


# ============================================================
# FEATURE EXPLANATIONS
# ============================================================

FEATURE_DESCRIPTIONS = {

    "aspect_ratio":
        "The word's width-to-height proportion differs from the reference words.",

    "ink_aspect_ratio":
        "The ink region has an unusual width-to-height proportion.",

    "ink_density":
        "The amount of ink relative to the word region differs from the reference distribution.",

    "centroid_x_ratio":
        "The horizontal center of the ink is positioned differently within the word region.",

    "centroid_y_ratio":
        "The vertical center of the ink is positioned differently within the word region.",

    "largest_component_ratio":
        "The proportion of ink belonging to the largest connected component is unusual.",

    "horizontal_projection_std":
        "The distribution of ink across horizontal rows differs from the reference words.",

    "vertical_projection_std":
        "The distribution of ink across vertical columns differs from the reference words.",

    "upper_ink_ratio":
        "The amount of ink in the upper part of the word differs from the reference distribution.",

    "lower_ink_ratio":
        "The amount of ink in the lower part of the word differs from the reference distribution.",
}


# ============================================================
# FEATURES USED FOR EXPLANATION
# ============================================================

FEATURES = list(FEATURE_DESCRIPTIONS.keys())


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

print("=" * 60)
print("WORD ANOMALY EXPLANATION")
print("=" * 60)

print(f"Input file: {INPUT_CSV}")
print(f"Total words: {len(df)}")


# ============================================================
# SORT BY WORD ANOMALY SCORE
# ============================================================

df = df.sort_values(
    "word_anomaly_score",
    ascending=False
).reset_index(drop=True)

top_df = df.head(TOP_K).copy()


# ============================================================
# GENERATE EXPLANATION
# ============================================================

explanation_rows = []

for rank, row in top_df.iterrows():

    rank = rank + 1

    word = str(row["transcription"])
    score = float(row["word_anomaly_score"])

    feature_scores = []

    for feature in FEATURES:

        z_column = f"{feature}_z"

        if z_column not in df.columns:
            continue

        value = row[z_column]

        if pd.isna(value):
            continue

        feature_scores.append(
            (
                feature,
                float(value),
                abs(float(value))
            )
        )

    # --------------------------------------------------------
    # Sort by strongest deviation
    # --------------------------------------------------------

    feature_scores.sort(
        key=lambda x: x[2],
        reverse=True
    )

    top_features = feature_scores[:3]

    # --------------------------------------------------------
    # Build human-readable explanation
    # --------------------------------------------------------

    explanation_parts = []

    for feature, z_value, abs_z in top_features:

        direction = (
            "higher"
            if z_value > 0
            else "lower"
        )

        description = FEATURE_DESCRIPTIONS[feature]

        explanation_parts.append(
            f"{feature}: {direction} than reference "
            f"(z={z_value:.2f}). {description}"
        )

    explanation = " ".join(explanation_parts)

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    explanation_rows.append({

        "rank":
            rank,

        "word_id":
            row["word_id"],

        "transcription":
            word,

        "word_anomaly_score":
            round(score, 6),

        "main_feature":
            top_features[0][0]
            if len(top_features) > 0
            else "",

        "main_feature_z":
            round(top_features[0][1], 6)
            if len(top_features) > 0
            else "",

        "second_feature":
            top_features[1][0]
            if len(top_features) > 1
            else "",

        "second_feature_z":
            round(top_features[1][1], 6)
            if len(top_features) > 1
            else "",

        "third_feature":
            top_features[2][0]
            if len(top_features) > 2
            else "",

        "third_feature_z":
            round(top_features[2][1], 6)
            if len(top_features) > 2
            else "",

        "explanation":
            explanation,

    })


# ============================================================
# SAVE REPORT
# ============================================================

explanation_df = pd.DataFrame(explanation_rows)

explanation_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT REPORT
# ============================================================

print()
print("TOP WORD EXPLANATIONS")
print("=" * 60)

for _, row in explanation_df.iterrows():

    print()

    print(
        f"#{int(row['rank'])} "
        f"{row['transcription']} "
        f"(score={row['word_anomaly_score']:.3f})"
    )

    print(
        f"  1. {row['main_feature']} "
        f"(z={row['main_feature_z']:.2f})"
    )

    if row["second_feature"]:
        print(
            f"  2. {row['second_feature']} "
            f"(z={row['second_feature_z']:.2f})"
        )

    if row["third_feature"]:
        print(
            f"  3. {row['third_feature']} "
            f"(z={row['third_feature_z']:.2f})"
        )

    print(
        f"  Explanation: {row['explanation']}"
    )


print()
print("=" * 60)
print("EXPLANATION REPORT COMPLETED")
print("=" * 60)

print()
print("Saved to:")
print(OUTPUT_CSV)
