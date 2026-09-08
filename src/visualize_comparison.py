import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "metadata" / "master_anomaly_analysis.csv"
OUTPUT_DIR = BASE_DIR / "data" / "metadata"

OUTPUT_PLOT = OUTPUT_DIR / "resnet_vs_word_anomaly.png"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("ANOMALY COMPARISON VISUALIZATION")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

print(f"Samples loaded: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "sample_id",
    "resnet_anomaly_score",
    "mean_word_anomaly",
    "anomaly_agreement"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")


# ============================================================
# SCATTER PLOT
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["resnet_anomaly_score"],
    df["mean_word_anomaly"],
    alpha=0.55,
    s=35
)

plt.xlabel("ResNet Anomaly Score (0–100)")
plt.ylabel("Mean Word Anomaly")
plt.title("ResNet vs Word-Level Handwriting Anomaly")

plt.grid(alpha=0.3)

# ------------------------------------------------------------
# Label the top combined samples
# ------------------------------------------------------------

top_samples = df.nlargest(10, "combined_anomaly_score")

for _, row in top_samples.iterrows():

    plt.annotate(
        row["sample_id"],
        (
            row["resnet_anomaly_score"],
            row["mean_word_anomaly"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300,
    bbox_inches="tight"
)



print()
print("Visualization saved:")
print(OUTPUT_PLOT)

print()
print("DONE.")