from pathlib import Path
import re
import random

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IAM"
    / "images"
)

WORDS_METADATA = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "words_metadata.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "all_samples_word_anomaly.csv"
)


# ============================================================
# SETTINGS
# ============================================================

PADDING = 10

REFERENCE_SIZE = 10000

RANDOM_SEED = 42

# Same 9 features used in the refined p06-052 analysis
ANOMALY_FEATURES = [
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
# PUNCTUATION DETECTION
# ============================================================

def is_punctuation(text):
    """
    Returns True if the transcription is punctuation only.
    """

    text = str(text).strip()

    if not text:
        return True

    # Remove whitespace
    cleaned = re.sub(r"\s+", "", text)

    # Word must contain at least one alphanumeric character
    return not bool(re.search(r"[A-Za-z0-9]", cleaned))


# ============================================================
# IMAGE MAP
# ============================================================

def build_image_map():
    """
    Build:
        form_id -> image path

    once, instead of searching the image directory repeatedly.
    """

    print("\nBuilding image map...")

    image_map = {}

    image_files = list(IMAGE_ROOT.rglob("*.png"))

    for image_path in image_files:

        form_id = image_path.stem

        image_map[form_id] = image_path

    print(f"Images found: {len(image_files)}")

    return image_map


# ============================================================
# WORD CROP
# ============================================================

def crop_word(image, row):
    """
    Crop one IAM word using its exact metadata bounding box.
    """

    x = int(row["bbox_x"])
    y = int(row["bbox_y"])

    w = int(row["bbox_width"])
    h = int(row["bbox_height"])

    height, width = image.shape[:2]

    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)

    x2 = min(width, x + w + PADDING)
    y2 = min(height, y + h + PADDING)

    crop = image[y1:y2, x1:x2]

    return crop


# ============================================================
# WORD FEATURE EXTRACTION
# ============================================================

def extract_features(gray):
    """
    Extract the same 16 features used previously.

    Only 9 are used for anomaly scoring.
    """

    if gray is None or gray.size == 0:
        return None

    # Otsu threshold
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove tiny noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)

    min_component_area = 2

    component_areas = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_component_area:

            cleaned[labels == i] = 255
            component_areas.append(area)

    binary = cleaned

    # Dimensions
    h, w = binary.shape

    if h == 0 or w == 0:
        return None

    # Ink pixels
    ink_pixels = np.count_nonzero(binary)

    if ink_pixels == 0:
        return None

    # --------------------------------------------------------
    # Basic shape
    # --------------------------------------------------------

    aspect_ratio = w / max(h, 1)

    # Ink bounding box
    ys, xs = np.where(binary > 0)

    ink_width = xs.max() - xs.min() + 1
    ink_height = ys.max() - ys.min() + 1

    ink_aspect_ratio = (
        ink_width / max(ink_height, 1)
    )

    ink_density = (
        ink_pixels / float(h * w)
    )

    # --------------------------------------------------------
    # Centroid
    # --------------------------------------------------------

    centroid_x = xs.mean()
    centroid_y = ys.mean()

    centroid_x_ratio = (
        centroid_x / max(w, 1)
    )

    centroid_y_ratio = (
        centroid_y / max(h, 1)
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    component_count = len(component_areas)

    if component_count > 0:

        largest_component = max(component_areas)

        largest_component_ratio = (
            largest_component / ink_pixels
        )

    else:

        largest_component = 0
        largest_component_ratio = 0

    # --------------------------------------------------------
    # Projection profiles
    # --------------------------------------------------------

    horizontal_projection = (
        np.sum(binary > 0, axis=1)
    )

    vertical_projection = (
        np.sum(binary > 0, axis=0)
    )

    horizontal_projection_std = (
        float(np.std(horizontal_projection))
    )

    vertical_projection_std = (
        float(np.std(vertical_projection))
    )

    # --------------------------------------------------------
    # Upper / lower ink
    # --------------------------------------------------------

    midpoint = h // 2

    upper_pixels = np.count_nonzero(
        binary[:midpoint]
    )

    lower_pixels = np.count_nonzero(
        binary[midpoint:]
    )

    upper_ink_ratio = (
        upper_pixels / ink_pixels
    )

    lower_ink_ratio = (
        lower_pixels / ink_pixels
    )

    return {
        "crop_width": w,
        "crop_height": h,

        "aspect_ratio": aspect_ratio,

        "ink_width": ink_width,
        "ink_height": ink_height,
        "ink_aspect_ratio": ink_aspect_ratio,

        "ink_density": ink_density,

        "centroid_x_ratio": centroid_x_ratio,
        "centroid_y_ratio": centroid_y_ratio,

        "component_count": component_count,

        "largest_component": largest_component,

        "largest_component_ratio":
            largest_component_ratio,

        "horizontal_projection_std":
            horizontal_projection_std,

        "vertical_projection_std":
            vertical_projection_std,

        "upper_ink_ratio":
            upper_ink_ratio,

        "lower_ink_ratio":
            lower_ink_ratio,
    }


# ============================================================
# REFERENCE SAMPLING
# ============================================================

def prepare_reference_words(words_df, image_map):
    """
    Randomly select 10,000 valid handwriting words.

    This is the reference distribution against which
    individual target words are compared.
    """

    print("\nPreparing reference words...")

    valid = words_df[
        words_df["segmentation_status"].astype(str).str.lower()
        == "ok"
    ].copy()

    # Remove punctuation
    valid = valid[
        ~valid["transcription"].apply(is_punctuation)
    ]

    # Only forms whose images actually exist
    valid = valid[
        valid["form_id"].isin(image_map)
    ]

    print(
        f"Valid handwriting word regions: "
        f"{len(valid)}"
    )

    sample_size = min(
        REFERENCE_SIZE,
        len(valid)
    )

    reference_df = valid.sample(
        n=sample_size,
        random_state=RANDOM_SEED
    )

    print(
        f"Using {sample_size} word regions "
        f"for reference statistics."
    )

    return reference_df


# ============================================================
# FEATURE EXTRACTION FOR REFERENCE
# ============================================================

def calculate_reference_statistics(
    reference_df,
    image_map
):

    print("\nExtracting reference features...")
    print("-" * 60)

    features = []

    skipped = 0

    current_form = None
    image = None

    for index, row in enumerate(
        reference_df.itertuples(index=False),
        start=1
    ):

        form_id = row.form_id

        # Load image only when form changes
        if form_id != current_form:

            image_path = image_map.get(form_id)

            if image_path is None:

                skipped += 1
                continue

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            current_form = form_id

        if image is None:

            skipped += 1
            continue

        row_dict = row._asdict()

        crop = crop_word(
            image,
            row_dict
        )

        feature_dict = extract_features(
            crop
        )

        if feature_dict is None:

            skipped += 1
            continue

        features.append(feature_dict)

        if index % 1000 == 0:

            print(
                f"Processed reference words: "
                f"{index}/{len(reference_df)}"
            )

    feature_df = pd.DataFrame(features)

    print(
        f"\nReference words successfully processed: "
        f"{len(feature_df)}"
    )

    print(
        f"Skipped: {skipped}"
    )

    return feature_df


# ============================================================
# ROBUST STATISTICS
# ============================================================

def calculate_robust_stats(feature_df):

    medians = {}

    mads = {}

    for feature in ANOMALY_FEATURES:

        values = pd.to_numeric(
            feature_df[feature],
            errors="coerce"
        ).dropna()

        median = values.median()

        mad = np.median(
            np.abs(values - median)
        )

        # Prevent division by zero
        if mad < 1e-8:
            mad = 1e-8

        medians[feature] = median
        mads[feature] = mad

    return medians, mads


# ============================================================
# WORD ANOMALY SCORE
# ============================================================

def calculate_word_anomaly(
    feature_dict,
    medians,
    mads
):

    z_scores = {}

    for feature in ANOMALY_FEATURES:

        value = float(
            feature_dict[feature]
        )

        robust_z = (
            0.6745
            * (value - medians[feature])
            / mads[feature]
        )

        z_scores[feature] = robust_z

    # Mean absolute robust z-score
    score = np.mean(
        [
            abs(z_scores[f])
            for f in ANOMALY_FEATURES
        ]
    )

    # Strongest contributor
    main_feature = max(
        ANOMALY_FEATURES,
        key=lambda f: abs(z_scores[f])
    )

    return (
        float(score),
        main_feature,
        float(z_scores[main_feature]),
        z_scores
    )


# ============================================================
# PROCESS ALL WORDS
# ============================================================

def process_all_words(
    words_df,
    image_map,
    medians,
    mads
):

    print("\n")
    print("=" * 60)
    print("PROCESSING ALL IAM WORDS")
    print("=" * 60)

    valid = words_df[
        words_df["segmentation_status"].astype(str).str.lower()
        == "ok"
    ].copy()

    valid = valid[
        ~valid["transcription"].apply(is_punctuation)
    ]

    valid = valid[
        valid["form_id"].isin(image_map)
    ]

    print(
        f"\nTotal valid handwriting words: "
        f"{len(valid)}"
    )

    results = []

    current_form = None
    image = None

    total = len(valid)

    for index, row in enumerate(
        valid.itertuples(index=False),
        start=1
    ):

        form_id = row.form_id

        # ----------------------------------------------------
        # Load each image only once
        # ----------------------------------------------------

        if form_id != current_form:

            image_path = image_map.get(
                form_id
            )

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            current_form = form_id

        if image is None:

            continue

        row_dict = row._asdict()

        crop = crop_word(
            image,
            row_dict
        )

        feature_dict = extract_features(
            crop
        )

        if feature_dict is None:

            continue

        (
            score,
            main_feature,
            main_z,
            z_scores
        ) = calculate_word_anomaly(
            feature_dict,
            medians,
            mads
        )

        result = {
            "word_id": row.word_id,
            "form_id": row.form_id,
            "line_id": row.line_id,
            "transcription": row.transcription,

            "bbox_x": row.bbox_x,
            "bbox_y": row.bbox_y,
            "bbox_width": row.bbox_width,
            "bbox_height": row.bbox_height,

            "word_anomaly_score": score,

            "main_anomaly_feature":
                main_feature,

            "main_feature_z":
                main_z,
        }

        # Add extracted features
        result.update(
            feature_dict
        )

        # Add z-scores
        for feature in ANOMALY_FEATURES:

            result[
                feature + "_z"
            ] = z_scores[feature]

        results.append(result)

        # Progress every 5000 words
        if index % 5000 == 0:

            percent = (
                index / total
            ) * 100

            print(
                f"Processed: "
                f"{index}/{total} "
                f"({percent:.1f}%)"
            )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FAST ALL-SAMPLE WORD ANOMALY ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not WORDS_METADATA.exists():

        print("\nERROR:")
        print(
            f"Missing metadata file:\n"
            f"{WORDS_METADATA}"
        )

        return

    if not IMAGE_ROOT.exists():

        print("\nERROR:")
        print(
            f"Missing image directory:\n"
            f"{IMAGE_ROOT}"
        )

        return

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print("\nLoading IAM word metadata...")

    words_df = pd.read_csv(
        WORDS_METADATA
    )

    print(
        f"Metadata rows: {len(words_df)}"
    )

    # --------------------------------------------------------
    # Image map
    # --------------------------------------------------------

    image_map = build_image_map()

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference_df = prepare_reference_words(
        words_df,
        image_map
    )

    reference_features = (
        calculate_reference_statistics(
            reference_df,
            image_map
        )
    )

    if len(reference_features) == 0:

        print(
            "\nERROR: Could not extract "
            "reference features."
        )

        return

    # --------------------------------------------------------
    # Robust statistics
    # --------------------------------------------------------

    print("\nCalculating reference statistics...")

    medians, mads = calculate_robust_stats(
        reference_features
    )

    print("\nReference statistics:")
    print("-" * 60)

    for feature in ANOMALY_FEATURES:

        print(
            f"{feature:<32}"
            f"median={medians[feature]:.5f}  "
            f"MAD={mads[feature]:.5f}"
        )

    # --------------------------------------------------------
    # Process every word
    # --------------------------------------------------------

    all_results = process_all_words(
        words_df,
        image_map,
        medians,
        mads
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"\nTotal word results: "
        f"{len(all_results)}"
    )

    print(
        f"Unique samples/forms: "
        f"{all_results['form_id'].nunique()}"
    )

    print(
        f"\nOutput:"
    )

    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Quick top anomalies
    # --------------------------------------------------------

    print("\nTOP 20 WORD ANOMALIES")
    print("-" * 60)

    top = (
        all_results
        .sort_values(
            "word_anomaly_score",
            ascending=False
        )
        .head(20)
    )

    print(
        top[
            [
                "form_id",
                "transcription",
                "word_anomaly_score",
                "main_anomaly_feature",
                "main_feature_z",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()