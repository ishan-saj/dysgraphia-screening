from pathlib import Path
import cv2
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_anomalies"
    / "synthetic_anomalies.csv"
)


# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("=" * 45)
print("Synthetic Dataset Validation")
print("=" * 45)

print(f"CSV samples: {len(df)}")


# --------------------------------------------------
# CHECK FILES AND MASKS
# --------------------------------------------------

valid_samples = 0
invalid_samples = 0

mask_pixels = []

for _, row in df.iterrows():

    original_path = Path(row["source_word"])
    synthetic_path = Path(row["synthetic_image"])
    mask_path = Path(row["ground_truth_mask"])

    original = cv2.imread(
        str(original_path),
        cv2.IMREAD_GRAYSCALE
    )

    synthetic = cv2.imread(
        str(synthetic_path),
        cv2.IMREAD_GRAYSCALE
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        invalid_samples += 1
        continue

    if synthetic is None:
        invalid_samples += 1
        continue

    if mask is None:
        invalid_samples += 1
        continue

    # Check dimensions
    if (
        original.shape != synthetic.shape
        or original.shape != mask.shape
    ):
        print(
            f"Dimension mismatch: {row['sample_id']}"
        )
        invalid_samples += 1
        continue

    # Check mask contains pixels
    pixels = np.sum(mask > 0)

    if pixels == 0:
        print(
            f"Empty mask: {row['sample_id']}"
        )
        invalid_samples += 1
        continue

    mask_pixels.append(pixels)

    valid_samples += 1


# --------------------------------------------------
# ANOMALY TYPE COUNTS
# --------------------------------------------------

print()
print("Anomaly types:")

print(
    df["anomaly_type"]
    .value_counts()
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("=" * 45)
print("Validation Results")
print("=" * 45)

print(f"Valid samples   : {valid_samples}")
print(f"Invalid samples : {invalid_samples}")

if mask_pixels:
    print(
        f"Minimum mask pixels : {min(mask_pixels)}"
    )

    print(
        f"Maximum mask pixels : {max(mask_pixels)}"
    )

    print(
        f"Average mask pixels : "
        f"{np.mean(mask_pixels):.1f}"
    )

print()

if invalid_samples == 0:
    print("STATUS: PASS")
    print("All synthetic samples are valid.")
else:
    print("STATUS: WARNING")
    print("Some samples need investigation.")