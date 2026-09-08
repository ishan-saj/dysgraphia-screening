from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FORM_ID = "p06-052"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / f"{FORM_ID}_word_metadata.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / f"{FORM_ID}_word_features.csv"
)


# ============================================================
# CREATE INK MASK
# ============================================================

def create_ink_mask(image):
    """
    Convert grayscale handwriting crop into a binary ink mask.

    White pixels = handwriting
    Black pixels = background
    """

    # Otsu threshold
    _, mask = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove tiny isolated noise
    kernel = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


# ============================================================
# CONNECTED COMPONENT FEATURES
# ============================================================

def get_component_features(mask):
    """
    Calculate connected-component statistics.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    components = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # Ignore extremely tiny noise
        if area >= 3:
            components.append(area)

    if not components:
        return 0, 0.0, 0.0

    components = np.array(components, dtype=np.float32)

    component_count = len(components)

    largest_component = float(np.max(components))

    total_component_area = float(np.sum(components))

    largest_component_ratio = (
        largest_component / total_component_area
        if total_component_area > 0
        else 0.0
    )

    return (
        component_count,
        largest_component,
        largest_component_ratio
    )


# ============================================================
# WORD FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not load image: {image_path}"
        )

    height, width = image.shape

    mask = create_ink_mask(image)

    # --------------------------------------------------------
    # Basic geometry
    # --------------------------------------------------------

    aspect_ratio = (
        width / height
        if height > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Ink density
    # --------------------------------------------------------

    ink_pixels = np.count_nonzero(mask)

    total_pixels = height * width

    ink_density = (
        ink_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Ink bounding box
    # --------------------------------------------------------

    ys, xs = np.where(mask > 0)

    if len(xs) > 0:

        ink_width = int(xs.max() - xs.min() + 1)
        ink_height = int(ys.max() - ys.min() + 1)

        ink_aspect_ratio = (
            ink_width / ink_height
            if ink_height > 0
            else 0.0
        )

        # Ink centroid
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))

        normalized_centroid_x = (
            centroid_x / width
            if width > 0
            else 0.0
        )

        normalized_centroid_y = (
            centroid_y / height
            if height > 0
            else 0.0
        )

    else:

        ink_width = 0
        ink_height = 0
        ink_aspect_ratio = 0.0

        normalized_centroid_x = 0.0
        normalized_centroid_y = 0.0

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    (
        component_count,
        largest_component,
        largest_component_ratio
    ) = get_component_features(mask)

    # --------------------------------------------------------
    # Horizontal ink distribution
    # --------------------------------------------------------

    horizontal_projection = np.sum(
        mask > 0,
        axis=1
    )

    vertical_projection = np.sum(
        mask > 0,
        axis=0
    )

    horizontal_projection_std = float(
        np.std(horizontal_projection)
    )

    vertical_projection_std = float(
        np.std(vertical_projection)
    )

    # --------------------------------------------------------
    # Upper / lower ink distribution
    # --------------------------------------------------------

    middle_y = height // 2

    upper_ink = np.count_nonzero(
        mask[:middle_y]
    )

    lower_ink = np.count_nonzero(
        mask[middle_y:]
    )

    total_vertical_ink = upper_ink + lower_ink

    upper_ink_ratio = (
        upper_ink / total_vertical_ink
        if total_vertical_ink > 0
        else 0.0
    )

    lower_ink_ratio = (
        lower_ink / total_vertical_ink
        if total_vertical_ink > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Return features
    # --------------------------------------------------------

    return {
        "crop_width": width,
        "crop_height": height,
        "aspect_ratio": aspect_ratio,

        "ink_width": ink_width,
        "ink_height": ink_height,
        "ink_aspect_ratio": ink_aspect_ratio,

        "ink_density": ink_density,

        "centroid_x_ratio": normalized_centroid_x,
        "centroid_y_ratio": normalized_centroid_y,

        "component_count": component_count,
        "largest_component": largest_component,
        "largest_component_ratio": largest_component_ratio,

        "horizontal_projection_std": horizontal_projection_std,
        "vertical_projection_std": vertical_projection_std,

        "upper_ink_ratio": upper_ink_ratio,
        "lower_ink_ratio": lower_ink_ratio,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("WORD-LEVEL HANDWRITING FEATURE EXTRACTION")
    print("=" * 60)

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            f"Could not find:\n{METADATA_PATH}"
        )

    metadata = pd.read_csv(METADATA_PATH)

    # Only real words, not punctuation
    metadata = metadata[
        metadata["is_punctuation"] == False
    ].copy()

    print(
        f"\nWords to analyze: {len(metadata)}"
    )

    results = []

    for index, row in metadata.iterrows():

        word = str(row["transcription"])

        crop_path = Path(
            row["crop_path"]
        )

        if not crop_path.exists():

            print(
                f"WARNING: Missing crop: {crop_path}"
            )

            continue

        try:

            features = extract_features(
                crop_path
            )

        except Exception as e:

            print(
                f"ERROR processing {word}: {e}"
            )

            continue

        result = {
            "word_id": row["word_id"],
            "form_id": row["form_id"],
            "line_id": row["line_id"],
            "transcription": word,
        }

        result.update(features)

        results.append(result)

        print(
            f"{len(results):02d} | "
            f"{word:<15} | "
            f"width={features['crop_width']:4d} | "
            f"height={features['crop_height']:3d} | "
            f"ink_density={features['ink_density']:.4f} | "
            f"components={features['component_count']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    print(
        f"Words analyzed: {len(result_df)}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    print("\nFeatures generated:")

    for column in result_df.columns:

        if column not in [
            "word_id",
            "form_id",
            "line_id",
            "transcription"
        ]:
            print(f"  - {column}")


if __name__ == "__main__":
    main()

