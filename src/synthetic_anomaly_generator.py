import os
import csv
import random
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# Project root:
# dysgraphia-screening/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Existing normal word crops
WORD_CROPS_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "word_crops"
)

# New synthetic anomaly dataset
OUTPUT_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_anomalies"
    / "images"
)

OUTPUT_MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_anomalies"
    / "masks"
)

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_anomalies"
    / "synthetic_anomalies.csv"
)

# Test with 20 first
NUM_SYNTHETIC_SAMPLES = 1000

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
}

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND WORD CROPS
# ============================================================

def find_word_images():

    images = []

    for path in WORD_CROPS_DIR.rglob("*"):

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return images


# ============================================================
# ANOMALY 1: STROKE THICKENING
# ============================================================
def thicken_region(image, mask):
    """
    Make handwriting strokes thicker inside the anomaly region.
    For black-on-white images, erosion makes black strokes thicker.
    """

    kernel_size = random.choice([2, 3, 4])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    eroded = cv2.erode(
        image,
        kernel,
        iterations=1
    )

    result = image.copy()
    result[mask > 0] = eroded[mask > 0]

    return result


def thin_region(image, mask):
    """
    Make handwriting strokes thinner inside the anomaly region.
    For black-on-white images, dilation makes black strokes thinner.
    """

    kernel_size = random.choice([2, 3])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated = cv2.dilate(
        image,
        kernel,
        iterations=1
    )

    result = image.copy()
    result[mask > 0] = dilated[mask > 0]

    return result

# ============================================================
# ANOMALY 3: LOCAL BLUR
# ============================================================

def blur_region(image, mask):

    blurred = cv2.GaussianBlur(
        image,
        (7, 7),
        0
    )

    result = image.copy()

    result[mask > 0] = blurred[mask > 0]

    return result


# ============================================================
# ANOMALY 4: STROKE DROPOUT
# ============================================================

def dropout_region(image, mask):

    result = image.copy()

    # Make selected region white
    result[mask > 0] = 255

    return result


# ============================================================
# CREATE RANDOM LOCAL REGION
# ============================================================

def create_random_mask(image):
    """
    Create a random rectangular anomaly mask that overlaps
    actual handwriting/ink pixels.

    The input word image is assumed to have:
        black/dark pixels = handwriting
        white/light pixels = background
    """

    h, w = image.shape

    # Find handwriting/ink pixels
    ink = image < 200

    # If there is no detectable ink, fall back to a random region
    ys, xs = np.where(ink)

    mask = np.zeros((h, w), dtype=np.uint8)

    if len(xs) == 0:
        region_width = max(5, int(w * 0.20))
        region_height = max(5, int(h * 0.40))

        x = random.randint(
            0,
            max(0, w - region_width)
        )

        y = random.randint(
            0,
            max(0, h - region_height)
        )

        mask[
            y:y + region_height,
            x:x + region_width
        ] = 255

        return mask

    # --------------------------------------------------------
    # Select a point that is actually inside handwriting
    # --------------------------------------------------------

    index = random.randint(0, len(xs) - 1)

    center_x = xs[index]
    center_y = ys[index]

    # --------------------------------------------------------
    # Random anomaly size
    # --------------------------------------------------------

    region_width = random.randint(
        max(5, int(w * 0.10)),
        max(6, int(w * 0.30))
    )

    region_height = random.randint(
        max(5, int(h * 0.25)),
        max(6, int(h * 0.70))
    )

    # --------------------------------------------------------
    # Center region around an actual ink pixel
    # --------------------------------------------------------

    x1 = center_x - region_width // 2
    y1 = center_y - region_height // 2

    x1 = max(0, min(x1, w - region_width))
    y1 = max(0, min(y1, h - region_height))

    x2 = x1 + region_width
    y2 = y1 + region_height

    mask[y1:y2, x1:x2] = 255

    # --------------------------------------------------------
    # Make sure the mask actually touches handwriting
    # --------------------------------------------------------

    overlap = np.logical_and(
        mask > 0,
        ink
    )

    if not np.any(overlap):
        # Expand around the selected ink point
        radius_x = max(3, region_width // 2)
        radius_y = max(3, region_height // 2)

        x1 = max(0, center_x - radius_x)
        x2 = min(w, center_x + radius_x)

        y1 = max(0, center_y - radius_y)
        y2 = min(h, center_y + radius_y)

        mask[y1:y2, x1:x2] = 255

    return mask


# ============================================================
# APPLY RANDOM ANOMALY
# ============================================================

def create_anomaly(image):

    mask = create_random_mask(image)

    anomaly_type = random.choice([
        "thickening",
        "thinning",
        "blur",
        "dropout"
    ])

    if anomaly_type == "thickening":

        result = thicken_region(
            image,
            mask
        )

    elif anomaly_type == "thinning":

        result = thin_region(
            image,
            mask
        )

    elif anomaly_type == "blur":

        result = blur_region(
            image,
            mask
        )

    else:

        result = dropout_region(
            image,
            mask
        )

    return result, mask, anomaly_type


# ============================================================
# MAIN
# ============================================================

def main():

    word_images = find_word_images()

    print(
        f"Found {len(word_images)} word images."
    )

    if len(word_images) == 0:

        print(
            "\nERROR: No word crops found."
        )

        print(
            f"Checked: {WORD_CROPS_DIR}"
        )

        return

    print(
        f"Generating {NUM_SYNTHETIC_SAMPLES} "
        "synthetic anomalies..."
    )

    csv_rows = []

    for i in range(NUM_SYNTHETIC_SAMPLES):

        # Random normal word
        source_path = random.choice(
            word_images
        )

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        # Ignore extremely small images
        if image.shape[0] < 10 or image.shape[1] < 10:
            continue

        # Generate anomaly
        synthetic_image, mask, anomaly_type = \
            create_anomaly(image)

        sample_id = f"synthetic_{i:05d}"

        image_path = (
            OUTPUT_IMAGE_DIR /
            f"{sample_id}.png"
        )

        mask_path = (
            OUTPUT_MASK_DIR /
            f"{sample_id}_mask.png"
        )

        # Save
        cv2.imwrite(
            str(image_path),
            synthetic_image
        )

        cv2.imwrite(
            str(mask_path),
            mask
        )

        csv_rows.append([
            sample_id,
            str(source_path),
            str(image_path),
            str(mask_path),
            anomaly_type,
            1
        ])

        if (i + 1) % 100 == 0:

            print(
                f"Generated {i + 1}/"
                f"{NUM_SYNTHETIC_SAMPLES}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "sample_id",
            "source_word",
            "synthetic_image",
            "ground_truth_mask",
            "anomaly_type",
            "label"
        ])

        writer.writerows(csv_rows)

    print("\n===================================")
    print("Synthetic anomaly generation done")
    print("===================================")

    print(
        f"Images: {OUTPUT_IMAGE_DIR}"
    )

    print(
        f"Masks:  {OUTPUT_MASK_DIR}"
    )

    print(
        f"CSV:    {CSV_PATH}"
    )

    print(
        f"Samples generated: {len(csv_rows)}"
    )


if __name__ == "__main__":
    main()