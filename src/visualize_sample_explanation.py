import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_ID = "a01-063"

BASE_DIR = Path(__file__).resolve().parent.parent
METADATA_DIR = BASE_DIR / "data" / "metadata"
IMAGE_DIR = BASE_DIR / "data" / "raw" / "IAM" / "images"

EXPLANATION_FILE = METADATA_DIR / f"{SAMPLE_ID}_explanation.csv"

OUTPUT_FILE = METADATA_DIR / f"{SAMPLE_ID}_word_explanation.png"


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(sample_id):

    matches = list(IMAGE_DIR.rglob(f"{sample_id}.png"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find image for {sample_id}"
        )

    return matches[0]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print(f"VISUAL WORD EXPLANATION: {SAMPLE_ID}")
print("=" * 70)

df = pd.read_csv(EXPLANATION_FILE)

image_path = find_image(SAMPLE_ID)

print()
print(f"Image: {image_path}")
print(f"Anomalous words: {len(df)}")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Could not read image.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# DRAW WORD BOXES
# ============================================================

for _, row in df.iterrows():

    x = int(row["bbox_x"])
    y = int(row["bbox_y"])
    w = int(row["bbox_width"])
    h = int(row["bbox_height"])

    word = str(row["transcription"])
    score = float(row["word_anomaly_score"])

    cv2.rectangle(
        image_rgb,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        4
    )

    label = f"{word} ({score:.2f})"

    cv2.putText(
        image_rgb,
        label,
        (x, max(30, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(18, 12))

plt.imshow(image_rgb)

plt.title(
    f"{SAMPLE_ID} - Top 10 Word-Level Anomalies",
    fontsize=16
)

plt.axis("off")

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("Visualization saved:")
print(OUTPUT_FILE)

print()
print("DONE.")