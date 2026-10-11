from pathlib import Path
import argparse
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM" / "images"
XML_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM" / "xml"
WORDS_METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "words_metadata.csv"
PAIRED_CSV = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "paired_clean_vs_synthetic.csv"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "synthetic_form_anomalies"
IMAGE_OUTPUT = OUTPUT_ROOT / "images"
MASK_OUTPUT = OUTPUT_ROOT / "masks"
CSV_OUTPUT = OUTPUT_ROOT / "synthetic_form_anomalies.csv"


# ============================================================
# SETTINGS
# ============================================================

NUM_SAMPLES = 1000
RANDOM_SEED = 42

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

IMAGE_OUTPUT.mkdir(parents=True, exist_ok=True)
MASK_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND IAM FORM IMAGES
# ============================================================

def find_images():
    images = []
    for extension in IMAGE_EXTENSIONS:
        images.extend(IMAGE_ROOT.rglob(extension))
    return sorted(images)


# ============================================================
# CREATE HANDWRITING MASK (XML OR WORDS_METADATA.CSV)
# ============================================================

def create_handwriting_mask(image, form_id, xml_path=None, word_boxes_by_form=None):
    h, w = image.shape
    binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    handwriting_mask = np.zeros((h, w), dtype=np.uint8)

    if xml_path is not None and xml_path.exists():
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for cmp in root.iter("cmp"):
            x = int(cmp.attrib["x"])
            y = int(cmp.attrib["y"])
            width = int(cmp.attrib["width"])
            height = int(cmp.attrib["height"])

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + width), min(h, y + height)
            if x1 < x2 and y1 < y2:
                handwriting_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    handwriting_mask[y1:y2, x1:x2],
                    binary[y1:y2, x1:x2]
                )
    elif word_boxes_by_form is not None and form_id in word_boxes_by_form:
        for (x, y, width, height) in word_boxes_by_form[form_id]:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + width)), min(h, int(y + height))
            if x1 < x2 and y1 < y2:
                handwriting_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    handwriting_mask[y1:y2, x1:x2],
                    binary[y1:y2, x1:x2]
                )
    else:
        return handwriting_mask

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
    ys, xs = np.where(ink_mask > 0)
    mask = np.zeros((h, w), dtype=np.uint8)

    if len(xs) == 0:
        return mask

    index = random.randint(0, len(xs) - 1)
    center_x = xs[index]
    center_y = ys[index]

    region_width = random.randint(max(20, int(w * 0.03)), max(30, int(w * 0.12)))
    region_height = random.randint(max(20, int(h * 0.03)), max(30, int(h * 0.12)))

    x1 = max(0, min(center_x - region_width // 2, w - region_width))
    y1 = max(0, min(center_y - region_height // 2, h - region_height))
    x2 = min(w, x1 + region_width)
    y2 = min(h, y1 + region_height)

    mask[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, ink_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.bitwise_and(mask, ink_mask)

    return mask


# ============================================================
# ANOMALY OPERATIONS
# ============================================================

def apply_thickening(image, mask):
    result = image.copy()
    kernel = np.ones((3, 3), np.uint8)
    thickened = cv2.erode(image, kernel, iterations=1)
    result[mask > 0] = thickened[mask > 0]
    return result


def apply_thinning(image, mask):
    result = image.copy()
    kernel = np.ones((2, 2), np.uint8)
    thinned = cv2.dilate(image, kernel, iterations=1)
    blended = (0.65 * thinned + 0.35 * image).astype(np.uint8)
    result[mask > 0] = blended[mask > 0]
    return result


def apply_blur(image, mask):
    result = image.copy()
    blurred = cv2.GaussianBlur(image, (5, 5), 1.2)
    result[mask > 0] = blurred[mask > 0]
    return result


def apply_dropout(image, mask):
    result = image.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return result

    background_value = int(np.percentile(image, 90))
    dropout_ratio = random.uniform(0.30, 0.50)
    num_drop = int(len(xs) * dropout_ratio)
    indices = np.random.choice(len(xs), size=num_drop, replace=False)

    for idx in indices:
        y = ys[idx]
        x = xs[idx]
        faded = int(0.75 * background_value + 0.25 * int(result[y, x]))
        result[y, x] = np.clip(faded, 0, 255)

    return result


def apply_anomaly(image, mask, anomaly_type):
    if anomaly_type == "thickening":
        return apply_thickening(image, mask)
    if anomaly_type == "thinning":
        return apply_thinning(image, mask)
    if anomaly_type == "blur":
        return apply_blur(image, mask)
    if anomaly_type == "dropout":
        return apply_dropout(image, mask)
    return image


# ============================================================
# RESTORE / VERIFY EXISTING BENCHMARK DATASET
# ============================================================

def verify_or_restore_existing_dataset():
    """
    If the 1,000 synthetic form images, masks, and paired metadata already exist on disk,
    rebuild/validate synthetic_form_anomalies.csv directly from them so cached embeddings
    and benchmark scores remain 100% synchronized.
    """
    if not PAIRED_CSV.exists():
        return False

    paired_df = pd.read_csv(PAIRED_CSV)
    if len(paired_df) != NUM_SAMPLES:
        return False

    existing_images = list(IMAGE_OUTPUT.glob("synthetic_form_*.png"))
    existing_masks = list(MASK_OUTPUT.glob("synthetic_form_*_mask.png"))
    if len(existing_images) < NUM_SAMPLES or len(existing_masks) < NUM_SAMPLES:
        return False

    print(f"Found existing {NUM_SAMPLES} synthetic images and masks in {OUTPUT_ROOT}.")
    print("Synchronizing and validating synthetic_form_anomalies.csv...")

    records = []
    for _, row in paired_df.iterrows():
        sample_id = str(row["sample_id"])
        source_form = str(row["source_form"])
        anomaly_type = str(row["anomaly_type"])

        image_path = IMAGE_OUTPUT / f"{sample_id}.png"
        mask_path = MASK_OUTPUT / f"{sample_id}_mask.png"

        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_pixels = int(np.sum(mask_img > 0)) if mask_img is not None else 0

        records.append({
            "sample_id": sample_id,
            "source_form": source_form,
            "synthetic_image": str(image_path.relative_to(PROJECT_ROOT)),
            "ground_truth_mask": str(mask_path.relative_to(PROJECT_ROOT)),
            "anomaly_type": anomaly_type,
            "label": 1,
            "mask_pixels": mask_pixels,
        })

    df = pd.DataFrame(records)
    df.to_csv(CSV_OUTPUT, index=False)

    print()
    print("Synthetic form anomaly dataset verified and synchronized.")
    print(f"Images: {IMAGE_OUTPUT}")
    print(f"Masks:  {MASK_OUTPUT}")
    print(f"CSV:    {CSV_OUTPUT}")
    print(f"Samples verified: {len(df)}")
    print()
    print("Anomaly distribution:")
    print(df["anomaly_type"].value_counts())
    return True


# ============================================================
# MAIN GENERATION PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate or verify 1,000 synthetic form anomalies.")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of all 1,000 synthetic images and masks from scratch.",
    )
    args = parser.parse_args()

    if not args.force_regenerate:
        if verify_or_restore_existing_dataset():
            return

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    images = find_images()
    print(f"Found {len(images)} IAM form images.")

    if len(images) == 0:
        print("No IAM form images found.")
        return

    # Load word bounding boxes from words_metadata.csv if raw XMLs are not present
    word_boxes_by_form = {}
    if WORDS_METADATA_CSV.exists():
        print(f"Loading handwriting bounding boxes from {WORDS_METADATA_CSV.name}...")
        words_df = pd.read_csv(WORDS_METADATA_CSV)
        for form_id, group in words_df.groupby("form_id"):
            boxes = list(zip(
                group["bbox_x"].values,
                group["bbox_y"].values,
                group["bbox_width"].values,
                group["bbox_height"].values,
            ))
            word_boxes_by_form[str(form_id)] = boxes

    records = []
    attempts_total = 0
    max_total_attempts = NUM_SAMPLES * 5

    while len(records) < NUM_SAMPLES and attempts_total < max_total_attempts:
        attempts_total += 1
        source_path = random.choice(images)
        form_id = source_path.stem
        xml_path = XML_ROOT / f"{form_id}.xml"

        if not xml_path.exists() and form_id not in word_boxes_by_form:
            continue

        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        ink_mask = create_handwriting_mask(
            image,
            form_id=form_id,
            xml_path=xml_path if xml_path.exists() else None,
            word_boxes_by_form=word_boxes_by_form,
        )

        anomaly_mask = create_random_mask(ink_mask)
        attempts = 0
        while np.sum(anomaly_mask > 0) == 0 and attempts < 10:
            anomaly_mask = create_random_mask(ink_mask)
            attempts += 1

        if np.sum(anomaly_mask > 0) == 0:
            continue

        anomaly_type = random.choice(ANOMALY_TYPES)
        synthetic_image = apply_anomaly(image, anomaly_mask, anomaly_type)

        sample_idx = len(records) + 1
        sample_id = f"synthetic_form_{sample_idx:05d}"
        image_path = IMAGE_OUTPUT / f"{sample_id}.png"
        mask_path = MASK_OUTPUT / f"{sample_id}_mask.png"

        cv2.imwrite(str(image_path), synthetic_image)
        cv2.imwrite(str(mask_path), anomaly_mask)

        records.append({
            "sample_id": sample_id,
            "source_form": form_id,
            "synthetic_image": str(image_path.relative_to(PROJECT_ROOT)),
            "ground_truth_mask": str(mask_path.relative_to(PROJECT_ROOT)),
            "anomaly_type": anomaly_type,
            "label": 1,
            "mask_pixels": int(np.sum(anomaly_mask > 0)),
        })

        if len(records) % 100 == 0:
            print(f"Generated {len(records)}/{NUM_SAMPLES}")

    if len(records) == 0:
        print("Error: No synthetic samples could be generated. Preserving existing CSV if any.")
        return

    df = pd.DataFrame(records)
    df.to_csv(CSV_OUTPUT, index=False)

    print()
    print("Synthetic form anomaly generation done.")
    print(f"Images: {IMAGE_OUTPUT}")
    print(f"Masks: {MASK_OUTPUT}")
    print(f"CSV: {CSV_OUTPUT}")
    print(f"Samples generated: {len(df)}")
    print()
    print("Anomaly distribution:")
    print(df["anomaly_type"].value_counts())


if __name__ == "__main__":
    main()