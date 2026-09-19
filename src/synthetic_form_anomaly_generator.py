from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET


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
XML_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IAM"
    / "xml"
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
# CREATE HANDWRITING MASK USING IAM XML
# ============================================================

def create_handwriting_mask(image, xml_path):
    h, w = image.shape
    # Binary image:
    # black handwriting -> white mask
    binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    handwriting_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    # Read IAM XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # IAM <cmp> boxes correspond to handwritten components
    for cmp in root.iter("cmp"):

        x = int(cmp.attrib["x"])
        y = int(cmp.attrib["y"])
        width = int(cmp.attrib["width"])
        height = int(cmp.attrib["height"])

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + width)
        y2 = min(h, y + height)

        if x1 >= x2 or y1 >= y2:
            continue

        # Keep only actual dark pixels inside
        # the IAM handwritten component box
        handwriting_mask[y1:y2, x1:x2] = cv2.bitwise_or(
            handwriting_mask[y1:y2, x1:x2],
            binary[y1:y2, x1:x2]
        )

    # Remove tiny isolated noise
    kernel = np.ones((2, 2), np.uint8)

    handwriting_mask = cv2.morphologyEx(
        handwriting_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return handwriting_mask


# ============================================================
# CREATE RANDOM ANOMALY REGION
# ============================================================

def create_random_mask(ink_mask):

    h, w = ink_mask.shape

    # Find pixels that belong to handwriting
    ys, xs = np.where(
        ink_mask > 0
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    if len(xs) == 0:
        return mask

    # Pick a random actual handwriting pixel
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

    # Create rectangular candidate region
    mask[
        y1:y2,
        x1:x2
    ] = 255

    # Keep ONLY handwriting pixels
    mask = cv2.bitwise_and(
        mask,
        ink_mask
    )

    # Slightly expand handwriting pixels
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    # IMPORTANT:
    # Constrain again to handwriting.
    # This prevents dilation from entering
    # printed text near the handwriting.
    mask = cv2.bitwise_and(
        mask,
        ink_mask
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

    for i in range(NUM_SAMPLES):

        # ----------------------------------------------------
        # Select a random IAM form
        # ----------------------------------------------------

        source_path = random.choice(images)

        # Corresponding IAM XML annotation
        xml_path = XML_ROOT / f"{source_path.stem}.xml"

        if not xml_path.exists():

            print(
                f"Skipping {source_path.stem}: "
                f"XML not found."
            )

            continue

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            print(
                f"Skipping {source_path.stem}: "
                f"image could not be loaded."
            )

            continue

        # ----------------------------------------------------
        # Create handwriting-only mask
        # using IAM XML annotations
        # ----------------------------------------------------

        ink_mask = create_handwriting_mask(
            image,
            xml_path
        )

        # ----------------------------------------------------
        # Create random anomaly region
        # ONLY inside handwriting
        # ----------------------------------------------------

        anomaly_mask = create_random_mask(
            ink_mask
        )

        # ----------------------------------------------------
        # Retry if mask is empty
        # ----------------------------------------------------

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

            print(
                f"Skipping {source_path.stem}: "
                f"could not create anomaly mask."
            )

            continue

        # ----------------------------------------------------
        # Select anomaly type
        # ----------------------------------------------------

        anomaly_type = random.choice(
            ANOMALY_TYPES
        )

        # ----------------------------------------------------
        # Apply anomaly
        # ----------------------------------------------------

        synthetic_image = apply_anomaly(
            image,
            anomaly_mask,
            anomaly_type
        )

        # ----------------------------------------------------
        # Generate filenames
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Save synthetic image
        # ----------------------------------------------------

        cv2.imwrite(
            str(image_path),
            synthetic_image
        )

        # ----------------------------------------------------
        # Save ground-truth anomaly mask
        # ----------------------------------------------------

        cv2.imwrite(
            str(mask_path),
            anomaly_mask
        )

        # ----------------------------------------------------
        # Store metadata
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if len(records) % 100 == 0:

            print(
                f"Generated "
                f"{len(records)}/{NUM_SAMPLES}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

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