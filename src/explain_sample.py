import pandas as pd
from pathlib import Path
import subprocess
import sys


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_ID = "a01-063"

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "data" / "metadata"


# ============================================================
# FILES
# ============================================================

MASTER_FILE = METADATA_DIR / "master_anomaly_analysis.csv"
WORD_FILE = METADATA_DIR / "all_samples_word_anomaly.csv"

OUTPUT_FILE = METADATA_DIR / f"{SAMPLE_ID}_explanation.csv"


# ============================================================
# LOAD MASTER SAMPLE INFORMATION
# ============================================================

print("=" * 70)
print(f"EXPLAINABILITY ANALYSIS: {SAMPLE_ID}")
print("=" * 70)

master = pd.read_csv(MASTER_FILE)

sample = master[master["sample_id"] == SAMPLE_ID]

if sample.empty:
    raise ValueError(f"Sample {SAMPLE_ID} not found.")

sample = sample.iloc[0]

print()
print("SAMPLE-LEVEL RESULTS")
print("-" * 70)

print(f"Sample ID:              {SAMPLE_ID}")
print(f"ResNet anomaly:         {sample['resnet_anomaly_score']:.4f}")
print(f"Mean word anomaly:      {sample['mean_word_anomaly']:.4f}")
print(f"Maximum word anomaly:   {sample['maximum_word_anomaly']:.4f}")
print(f"Top-5 word anomaly:     {sample['top_5_mean_anomaly']:.4f}")
print(f"Combined anomaly:       {sample['combined_anomaly_score']:.4f}")
print(f"Agreement:              {sample['anomaly_agreement']}")


# ============================================================
# LOAD WORD-LEVEL DATA
# ============================================================

print()
print("Loading word-level anomaly data...")

words = pd.read_csv(WORD_FILE)

sample_words = words[words["form_id"] == SAMPLE_ID].copy()

if sample_words.empty:
    raise ValueError(f"No word-level data found for {SAMPLE_ID}.")


# ============================================================
# SORT WORDS BY ANOMALY
# ============================================================

sample_words = sample_words.sort_values(
    "word_anomaly_score",
    ascending=False
)


# ============================================================
# SELECT TOP 10
# ============================================================

top_words = sample_words.head(10).copy()

print()
print("TOP 10 ANOMALOUS WORDS")
print("-" * 70)

print(
    top_words[
        [
            "transcription",
            "word_anomaly_score",
            "main_anomaly_feature",
            "main_feature_z"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE EXPLANATION TABLE
# ============================================================

columns_to_save = [
    "word_id",
    "form_id",
    "line_id",
    "transcription",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "word_anomaly_score",
    "main_anomaly_feature",
    "main_feature_z",
    "aspect_ratio",
    "ink_aspect_ratio",
    "ink_density",
    "centroid_x_ratio",
    "centroid_y_ratio",
    "largest_component_ratio",
    "horizontal_projection_std",
    "vertical_projection_std",
    "lower_ink_ratio"
]

available_columns = [
    column for column in columns_to_save
    if column in top_words.columns
]

top_words[available_columns].to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("EXPLANATION FILE CREATED")
print("=" * 70)

print(OUTPUT_FILE)

print()
print("DONE.")