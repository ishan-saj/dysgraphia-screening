from pathlib import Path

import cv2
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM" / "images"

ANOMALY_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "p06-052_word_anomaly.csv"
)

TARGET_FORM = "p06-052"

TOP_N = 10

OUTPUT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / f"{TARGET_FORM}_top_word_anomalies.png"
)

OUTPUT_CROPS = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / f"{TARGET_FORM}_top_word_anomaly_crops"
)


# ============================================================
# FIND IMAGE
# ============================================================

print("=" * 60)
print("WORD ANOMALY VISUALIZATION")
print("=" * 60)

image_paths = list(IMAGE_ROOT.rglob(f"{TARGET_FORM}.*"))

if not image_paths:

    raise FileNotFoundError(
        f"Could not find image for {TARGET_FORM}"
    )

image_path = image_paths[0]

print(f"\nImage:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(image_path),
    cv2.IMREAD_COLOR
)

if image is None:

    raise RuntimeError(
        f"Could not load image:\n{image_path}"
    )

print(
    f"Image size: "
    f"{image.shape[1]} x {image.shape[0]}"
)


# ============================================================
# LOAD ANOMALY RESULTS
# ============================================================

df = pd.read_csv(ANOMALY_CSV)

df = df.sort_values(
    "word_anomaly_score",
    ascending=False
).reset_index(drop=True)

top_words = df.head(TOP_N).copy()

print(
    f"\nTop {len(top_words)} anomalous words:"
)

print(
    top_words[
        [
            "transcription",
            "word_anomaly_score",
            "main_anomaly_feature",
            "main_feature_z",
        ]
    ].to_string(index=False)
)


# ============================================================
# OUTPUT CROPS FOLDER
# ============================================================

OUTPUT_CROPS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DRAW WORD BOXES
# ============================================================

for rank, (_, row) in enumerate(
    top_words.iterrows(),
    start=1
):

    x = int(row["bbox_x"])
    y = int(row["bbox_y"])

    w = int(row["bbox_width"])
    h = int(row["bbox_height"])

    x2 = x + w
    y2 = y + h

    word = str(row["transcription"])

    score = float(
        row["word_anomaly_score"]
    )

    feature = str(
        row["main_anomaly_feature"]
    )

    # --------------------------------------------------------
    # Draw rectangle
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (x, y),
        (x2, y2),
        (0, 0, 255),
        5
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = (
        f"#{rank} {word} "
        f"({score:.2f})"
    )

    cv2.putText(
        image,
        label,
        (x, max(y - 15, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Save individual crop
    # --------------------------------------------------------

    padding = 20

    crop_x1 = max(
        0,
        x - padding
    )

    crop_y1 = max(
        0,
        y - padding
    )

    crop_x2 = min(
        image.shape[1],
        x2 + padding
    )

    crop_y2 = min(
        image.shape[0],
        y2 + padding
    )

    # Read clean original image for crop
    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    crop = original[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    crop_filename = (
        f"{rank:02d}_"
        f"{word}_"
        f"score_{score:.2f}.png"
    )

    crop_path = (
        OUTPUT_CROPS
        / crop_filename
    )

    cv2.imwrite(
        str(crop_path),
        crop
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

cv2.imwrite(
    str(OUTPUT_IMAGE),
    image
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)

print(
    f"\nAnnotated image:\n"
    f"{OUTPUT_IMAGE}"
)

print(
    f"\nIndividual word crops:\n"
    f"{OUTPUT_CROPS}"
)