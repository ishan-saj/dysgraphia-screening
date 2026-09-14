from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IAM"
    / "images"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
)

IMAGE_OUTPUT = OUTPUT_ROOT / "images"
MASK_OUTPUT = OUTPUT_ROOT / "masks"
CSV_OUTPUT = OUTPUT_ROOT / "synthetic_form_anomalies.csv"


# ============================================================
# SETTINGS
# ============================================================

NUM_SAMPLES = 1000

ANOMALY_TYPES = [
    "thickening",
    "thinning",
    "blur",
    "dropout",
]

IMAGE_EXTENSIONS = [
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.bmp",
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

IMAGE_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

MASK_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND IAM FORM IMAGES
# ============================================================

def find_images():

    images = []

    for extension in IMAGE_EXTENSIONS:

        images.extend(
            IMAGE_ROOT.rglob(extension)
        )

    return sorted(images)


# ============================================================
# CREATE HANDWRITING MASK
# ============================================================

def create_handwriting_mask(image):

    # Adaptive threshold for handwriting
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    # Remove tiny noise
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    return cleaned


# ============================================================
# CREATE RANDOM ANOMALY REGION
# ============================================================

def create_random_mask(ink_mask):

    h, w = ink_mask.shape

    ys, xs = np.where(
        ink_mask > 0
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    if len(xs) == 0:
        return mask

    # Pick an actual handwriting pixel
    index = random.randint(
        0,
        len(xs) - 1
    )

    center_x = xs[index]
    center_y = ys[index]

    # Moderate region size
    region_width = random.randint(
        max(20, int(w * 0.03)),
        max(30, int(w * 0.12))
    )

    region_height = random.randint(
        max(20, int(h * 0.03)),
        max(30, int(h * 0.12))
    )

    x1 = center_x - region_width // 2
    y1 = center_y - region_height // 2

    x1 = max(
        0,
        min(
            x1,
            w - region_width
        )
    )

    y1 = max(
        0,
        min(
            y1,
            h - region_height
        )
    )

    x2 = min(
        w,
        x1 + region_width
    )

    y2 = min(
        h,
        y1 + region_height
    )

    mask[
        y1:y2,
        x1:x2
    ] = 255

    # Only keep anomaly region overlapping handwriting
    mask = cv2.bitwise_and(
        mask,
        ink_mask
    )

    # Slightly expand the region so the
    # ground-truth mask covers the modification
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    return mask


# ============================================================
# THICKEN HANDWRITING
# ============================================================

def thicken_region(image, mask):

    kernel_size = random.choice(
        [2, 3, 4]
    )

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    # Black handwriting on white background:
    # erosion expands dark pixels
    modified = cv2.erode(
        image,
        kernel,
        iterations=1
    )

    result = image.copy()

    result[mask > 0] = modified[mask > 0]

    return result


# ============================================================
# THIN HANDWRITING
# ============================================================

def thin_region(image, mask):

    kernel_size = random.choice(
        [2, 3]
    )

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    # Dilation reduces black handwriting
    modified = cv2.dilate(
        image,
        kernel,
        iterations=1
    )

    result = image.copy()

    result[mask > 0] = modified[mask > 0]

    return result


# ============================================================
# BLUR REGION
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
# DROPOUT REGION
# ============================================================

def dropout_region(image, mask):

    result = image.copy()

    # Replace affected handwriting with white
    result[mask > 0] = 255

    return result


# ============================================================
# APPLY ANOMALY
# ============================================================

def apply_anomaly(
    image,
    mask,
    anomaly_type
):

    if anomaly_type == "thickening":

        return thicken_region(
            image,
            mask
        )

    if anomaly_type == "thinning":

        return thin_region(
            image,
            mask
        )

    if anomaly_type == "blur":

        return blur_region(
            image,
            mask
        )

    if anomaly_type == "dropout":

        return dropout_region(
            image,
            mask
        )

    raise ValueError(
        f"Unknown anomaly type: {anomaly_type}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(42)

    images = find_images()

    print(
        f"Found {len(images)} IAM form images."
    )

    if not images:

        raise RuntimeError(
            "No IAM form images found."
        )

    records = []

    print(
        f"Generating {NUM_SAMPLES} "
        f"synthetic form anomalies..."
    )

    for i in range(
        NUM_SAMPLES
    ):

        source_path = random.choice(
            images
        )

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        ink_mask = create_handwriting_mask(
            image
        )

        anomaly_mask = create_random_mask(
            ink_mask
        )

        # Retry if mask is empty
        attempts = 0

        while (
            np.sum(anomaly_mask > 0) == 0
            and attempts < 10
        ):

            anomaly_mask = create_random_mask(
                ink_mask
            )

            attempts += 1

        if np.sum(anomaly_mask > 0) == 0:
            continue

        anomaly_type = random.choice(
            ANOMALY_TYPES
        )

        synthetic_image = apply_anomaly(
            image,
            anomaly_mask,
            anomaly_type
        )

        sample_id = (
            f"synthetic_form_{i + 1:05d}"
        )

        image_name = (
            f"{sample_id}.png"
        )

        mask_name = (
            f"{sample_id}_mask.png"
        )

        image_path = (
            IMAGE_OUTPUT / image_name
        )

        mask_path = (
            MASK_OUTPUT / mask_name
        )

        cv2.imwrite(
            str(image_path),
            synthetic_image
        )

        cv2.imwrite(
            str(mask_path),
            anomaly_mask
        )

        records.append({

            "sample_id":
                sample_id,

            "source_form":
                source_path.stem,

            "synthetic_image":
                str(
                    image_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "ground_truth_mask":
                str(
                    mask_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "anomaly_type":
                anomaly_type,

            "label":
                1,

            "mask_pixels":
                int(
                    np.sum(
                        anomaly_mask > 0
                    )
                )

        })

        if len(records) % 100 == 0:

            print(
                f"Generated "
                f"{len(records)}/{NUM_SAMPLES}"
            )

    df = pd.DataFrame(
        records
    )

    df.to_csv(
        CSV_OUTPUT,
        index=False
    )

    print()
    print(
        "Synthetic form anomaly generation done."
    )

    print(
        f"Images: {IMAGE_OUTPUT}"
    )

    print(
        f"Masks: {MASK_OUTPUT}"
    )

    print(
        f"CSV: {CSV_OUTPUT}"
    )

    print(
        f"Samples generated: {len(df)}"
    )

    if len(df) > 0:

        print()
        print(
            "Anomaly distribution:"
        )

        print(
            df["anomaly_type"]
            .value_counts()
        )


if __name__ == "__main__":
    main()