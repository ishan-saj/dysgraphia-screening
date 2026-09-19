import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM" / "images"
WORDS_METADATA = PROJECT_ROOT / "data" / "metadata" / "words_metadata.csv"

TARGET_FORM = "p06-052"

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / f"{TARGET_FORM}_word_anomaly.csv"
)

REFERENCE_SIZE = 10000
RANDOM_SEED = 42

PADDING = 10


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image, x, y, w, h):
    """
    Extract handwriting-shape features from one IAM word box.
    """

    height, width = image.shape[:2]

    # Add the SAME padding for reference and target words
    x1 = max(0, int(x) - PADDING)
    y1 = max(0, int(y) - PADDING)

    x2 = min(width, int(x + w) + PADDING)
    y2 = min(height, int(y + h) + PADDING)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Binary ink mask
    _, mask = cv2.threshold(
        crop,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    crop_h, crop_w = mask.shape

    if crop_h == 0 or crop_w == 0:
        return None

    ink_pixels = mask > 0

    # --------------------------------------------------------
    # Basic shape
    # --------------------------------------------------------

    aspect_ratio = crop_w / max(crop_h, 1)

    ink_ys, ink_xs = np.where(ink_pixels)

    if len(ink_xs) > 0:

        ink_width = ink_xs.max() - ink_xs.min() + 1
        ink_height = ink_ys.max() - ink_ys.min() + 1

        ink_aspect_ratio = (
            ink_width / max(ink_height, 1)
        )

        centroid_x_ratio = (
            np.mean(ink_xs) / max(crop_w, 1)
        )

        centroid_y_ratio = (
            np.mean(ink_ys) / max(crop_h, 1)
        )

    else:

        ink_width = 0
        ink_height = 0
        ink_aspect_ratio = 0
        centroid_x_ratio = 0.5
        centroid_y_ratio = 0.5

    # --------------------------------------------------------
    # Ink density
    # --------------------------------------------------------

    ink_density = np.mean(ink_pixels)

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    component_count = max(0, num_labels - 1)

    if component_count > 0:

        component_areas = stats[1:, cv2.CC_STAT_AREA]

        largest_component = np.max(component_areas)

        total_ink_area = np.sum(component_areas)

        largest_component_ratio = (
            largest_component / max(total_ink_area, 1)
        )

    else:

        largest_component_ratio = 0

    # --------------------------------------------------------
    # Projection features
    # --------------------------------------------------------

    horizontal_projection = np.sum(ink_pixels, axis=1)
    vertical_projection = np.sum(ink_pixels, axis=0)

    horizontal_projection_std = np.std(
        horizontal_projection / max(crop_w, 1)
    )

    vertical_projection_std = np.std(
        vertical_projection / max(crop_h, 1)
    )

    # --------------------------------------------------------
    # Upper/lower ink distribution
    # --------------------------------------------------------

    midpoint = crop_h // 2

    upper_ink = np.sum(
        ink_pixels[:midpoint]
    )

    lower_ink = np.sum(
        ink_pixels[midpoint:]
    )

    total_ink = upper_ink + lower_ink

    if total_ink > 0:
        upper_ink_ratio = upper_ink / total_ink
        lower_ink_ratio = lower_ink / total_ink
    else:
        upper_ink_ratio = 0.5
        lower_ink_ratio = 0.5

    return {
        "aspect_ratio": aspect_ratio,
        "ink_aspect_ratio": ink_aspect_ratio,
        "ink_density": ink_density,
        "centroid_x_ratio": centroid_x_ratio,
        "centroid_y_ratio": centroid_y_ratio,
        "largest_component_ratio": largest_component_ratio,
        "horizontal_projection_std": horizontal_projection_std,
        "vertical_projection_std": vertical_projection_std,
        "upper_ink_ratio": upper_ink_ratio,
        "lower_ink_ratio": lower_ink_ratio,
    }


# ============================================================
# PUNCTUATION CHECK
# ============================================================

def is_punctuation(text):
    """
    Identify punctuation regions so they are not used
    for handwriting anomaly analysis.
    """

    if not isinstance(text, str):
        return False

    text = text.strip()

    if text == "":
        return False

    return bool(re.fullmatch(r"[^\w\s]+", text))


# ============================================================
# BUILD IMAGE MAP
# ============================================================

print("=" * 60)
print("FAST DATASET-BASED WORD ANOMALY ANALYSIS")
print("=" * 60)

print("\nScanning IAM images once...")

image_map = {}

for image_path in IMAGE_ROOT.rglob("*"):

    if image_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:

        form_id = image_path.stem

        image_map[form_id] = image_path

print(f"Images found: {len(image_map)}")


# ============================================================
# LOAD WORD METADATA
# ============================================================

words_df = pd.read_csv(WORDS_METADATA)

valid_words = words_df[
    words_df["segmentation_status"] == "ok"
].copy()

# Remove punctuation from reference dataset
valid_words = valid_words[
    ~valid_words["transcription"].apply(is_punctuation)
].copy()

valid_words = valid_words.reset_index(drop=True)

print(f"Valid handwriting word regions: {len(valid_words)}")


# ============================================================
# SELECT 10,000 REFERENCE WORDS
# ============================================================

if len(valid_words) > REFERENCE_SIZE:

    reference_df = valid_words.sample(
        n=REFERENCE_SIZE,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

else:

    reference_df = valid_words.copy()

print(
    f"\nUsing {len(reference_df)} word regions "
    f"for reference statistics."
)

print(
    "The remaining IAM regions are NOT deleted "
    "and can be used later."
)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURE_NAMES = [
    "aspect_ratio",
    "ink_aspect_ratio",
    "ink_density",
    "centroid_x_ratio",
    "centroid_y_ratio",
    "largest_component_ratio",
    "horizontal_projection_std",
    "vertical_projection_std",
    "lower_ink_ratio",
]


# ============================================================
# BUILD REFERENCE FEATURES
# ============================================================

print("\nBuilding reference distribution...")

reference_features = []

processed = 0
skipped = 0

image_cache = {}


for _, row in reference_df.iterrows():

    form_id = row["form_id"]

    if form_id not in image_map:
        skipped += 1
        continue

        # Load each form image only once
    try:
        image = cv2.imread(
            str(image_map[form_id]),
            cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            skipped += 1
            continue
    except Exception:
        skipped += 1
        continue

    features = extract_features(
        image,
        row["bbox_x"],
        row["bbox_y"],
        row["bbox_width"],
        row["bbox_height"],
    )

    if features is not None:

        reference_features.append(features)

    processed += 1

    if processed % 500 == 0:
        print(
            f"  Processed {processed} / "
            f"{len(reference_df)} reference words..."
        )


reference_features_df = pd.DataFrame(
    reference_features
)

print(
    f"\nReference words successfully processed: "
    f"{len(reference_features_df)}"
)

print(f"Skipped: {skipped}")


# ============================================================
# ROBUST REFERENCE STATISTICS
# ============================================================

print("\nCalculating robust reference statistics...")

medians = reference_features_df[
    FEATURE_NAMES
].median()

mads = (
    reference_features_df[FEATURE_NAMES]
    .sub(medians)
    .abs()
    .median()
)

# Avoid division by zero
mads = mads.replace(0, 1e-6)


# ============================================================
# TARGET WORDS
# ============================================================

target_df = valid_words[
    valid_words["form_id"] == TARGET_FORM
].copy()

target_df = target_df.reset_index(drop=True)

print(
    f"\nTarget form: {TARGET_FORM}"
)

print(
    f"Target handwriting words: "
    f"{len(target_df)}"
)


# ============================================================
# ANALYZE TARGET WORDS
# ============================================================

target_image_path = image_map.get(TARGET_FORM)

if target_image_path is None:

    raise FileNotFoundError(
        f"Could not find image for {TARGET_FORM}"
    )


target_image = cv2.imread(
    str(target_image_path),
    cv2.IMREAD_GRAYSCALE
)

if target_image is None:

    raise RuntimeError(
        f"Could not load image: {target_image_path}"
    )


results = []


for _, row in target_df.iterrows():

    features = extract_features(
        target_image,
        row["bbox_x"],
        row["bbox_y"],
        row["bbox_width"],
        row["bbox_height"],
    )

    if features is None:
        continue

    # --------------------------------------------------------
    # Robust z-score for every feature
    # --------------------------------------------------------

    feature_scores = {}

    for feature in FEATURE_NAMES:

        robust_z = (
            abs(
                features[feature]
                - medians[feature]
            )
            / (1.4826 * mads[feature])
        )

        feature_scores[
            f"{feature}_z"
        ] = robust_z

    # --------------------------------------------------------
    # Overall word anomaly
    # --------------------------------------------------------

    z_values = list(
        feature_scores.values()
    )

    word_anomaly = float(
        np.mean(z_values)
    )

    # Largest contributing feature
    highest_feature = max(
        feature_scores,
        key=feature_scores.get
    )

    highest_feature_name = (
        highest_feature.replace("_z", "")
    )

    highest_feature_score = feature_scores[
        highest_feature
    ]

    result = {
        "word_id": row["word_id"],
        "form_id": row["form_id"],
        "line_id": row["line_id"],
        "transcription": row["transcription"],
        "bbox_x": row["bbox_x"],
        "bbox_y": row["bbox_y"],
        "bbox_width": row["bbox_width"],
        "bbox_height": row["bbox_height"],
        "word_anomaly_score": word_anomaly,
        "main_anomaly_feature": highest_feature_name,
        "main_feature_z": highest_feature_score,
    }

    result.update(features)
    result.update(feature_scores)

    results.append(result)


# ============================================================
# SORT AND SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "word_anomaly_score",
    ascending=False
).reset_index(drop=True)


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TOP UNUSUAL WORDS")
print("=" * 60)

print(
    results_df[
        [
            "transcription",
            "word_anomaly_score",
            "main_anomaly_feature",
            "main_feature_z",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)

print(
    f"\nReference size used: "
    f"{len(reference_features_df)} words"
)

print(
    f"Target words analyzed: "
    f"{len(results_df)}"
)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_CSV}"
)