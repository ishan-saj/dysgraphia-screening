from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_anomalies"
    / "synthetic_anomalies.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print(f"Found {len(df)} synthetic samples.")

# Show first 6 samples
samples = df.head(6)


# --------------------------------------------------
# DISPLAY SAMPLES
# --------------------------------------------------

for _, row in samples.iterrows():

    original_path = row["source_word"]
    synthetic_path = row["synthetic_image"]
    mask_path = row["ground_truth_mask"]

    anomaly_type = row["anomaly_type"]

    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    synthetic = cv2.imread(synthetic_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        print(f"Could not read original: {original_path}")
        continue

    if synthetic is None:
        print(f"Could not read synthetic: {synthetic_path}")
        continue

    if mask is None:
        print(f"Could not read mask: {mask_path}")
        continue

    # Create mask overlay
    overlay = original.copy()

    # Highlight masked region
    overlay[mask > 0] = 0

    # Display
    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(synthetic, cmap="gray")
    plt.title(f"Synthetic\n{anomaly_type}")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay, cmap="gray")
    plt.title("Mask Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()