from pathlib import Path
import argparse
import random
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM" / "images"
BBOX_CSV = PROJECT_ROOT / "data" / "metadata" / "form_handwriting_bboxes.csv"
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
# FIND IAM FORM IMAGES & BBOXES
# ============================================================

def find_images():
    images = {}
    for extension in IMAGE_EXTENSIONS:
        for p in IMAGE_ROOT.rglob(extension):
            images[p.stem] = p
    return images


def load_bbox_map():
    if not BBOX_CSV.exists():
        return {}
    df = pd.read_csv(BBOX_CSV)
    bbox_map = {}
    for _, r in df.iterrows():
        bbox_map[str(r['form_id'])] = (int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2']))
    return bbox_map


# ============================================================
# CREATE HANDWRITING-CONSTRAINED ANOMALY MASK
# ============================================================

def create_random_handwriting_mask(image, form_id, bbox_map, rng):
    """
    Creates an anomaly mask strictly inside the genuine cursive handwriting region.
    Never overlaps the printed form header / prompt text at the top.
    """
    h, w = image.shape
    bb = bbox_map.get(form_id, (320, 750, w - 100, h - 100))
    x1, y1, x2, y2 = bb
    if x1 < 0 or y1 < 0:
        x1, y1 = 320, 750
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    # Inverted Otsu threshold to locate dark handwriting ink pixels
    inv = 255.0 - image.astype(np.float32)
    _, otsu = cv2.threshold(inv.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Restrict STRICTLY to handwriting region
    hw_ink = np.zeros((h, w), dtype=np.uint8)
    hw_ink[y1:y2, x1:x2] = otsu[y1:y2, x1:x2]

    ys, xs = np.where(hw_ink > 0)
    if len(xs) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Pick a random center ink pixel strictly within handwriting
    idx = rng.randint(0, len(xs))
    cx, cy = xs[idx], ys[idx]

    # Patch size: covers a representative cluster of handwritten words (20-45% width, 12-30% height)
    hw_w = max(100, x2 - x1)
    hw_h = max(100, y2 - y1)
    reg_w = rng.randint(int(hw_w * 0.20), int(hw_w * 0.45))
    reg_h = rng.randint(int(hw_h * 0.12), int(hw_h * 0.30))

    rx1 = max(x1, min(cx - reg_w // 2, x2 - reg_w))
    ry1 = max(y1, min(cy - reg_h // 2, y2 - reg_h))
    rx2 = min(x2, rx1 + reg_w)
    ry2 = min(y2, ry1 + reg_h)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ry1:ry2, rx1:rx2] = 255
    # Intersect with handwriting ink
    mask = cv2.bitwise_and(mask, hw_ink)
    return mask


# ============================================================
# CLINICAL ANOMALY PERTURBATION OPERATIONS
# ============================================================

def apply_thickening(image, mask):
    """Muscular spasm and heavy downward pen pressure: smooth elliptical dilation."""
    result = image.copy()
    if np.sum(mask > 0) == 0:
        return result
    inv = 255.0 - image.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dil = cv2.dilate(inv.astype(np.uint8), kernel, iterations=1)
    dil = cv2.GaussianBlur(dil, (3, 3), 0.5)
    thick = 255.0 - dil.astype(np.float32)
    result[mask > 0] = np.clip(thick[mask > 0], 0, 255).astype(np.uint8)
    return result


def apply_thinning(image, mask):
    """Micrographia and paresis (faint, thin stroke): smooth elliptical erosion + fade."""
    result = image.copy()
    if np.sum(mask > 0) == 0:
        return result
    inv = 255.0 - image.astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(inv.astype(np.uint8), kernel, iterations=1)
    thinned = 255.0 - (eroded.astype(np.float32) * 0.65)
    result[mask > 0] = np.clip(thinned[mask > 0], 0, 255).astype(np.uint8)
    return result


def apply_blur(image, mask):
    """Neuromuscular tremor blur."""
    result = image.copy()
    if np.sum(mask > 0) == 0:
        return result
    blurred = cv2.GaussianBlur(image, (13, 13), 3.0)
    result[mask > 0] = blurred[mask > 0]
    return result


def apply_dropout(image, mask, rng):
    """Neuromuscular hesitation / pen-lift stroke fragmentation."""
    result = image.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return result
    n_breaks = max(10, int(len(xs) * 0.006))
    chosen = rng.choice(len(xs), min(n_breaks, len(xs)), replace=False)
    drop_mask = np.ones_like(mask, dtype=bool)
    for i in chosen:
        cv2.ellipse(drop_mask, (xs[i], ys[i]), (9, 6), rng.randint(0, 180), 0, 360, False, -1)
    paper_bg = int(np.percentile(image[mask == 0], 70))
    result[~drop_mask & (mask > 0)] = paper_bg
    result = cv2.medianBlur(result, 3)
    return result


def apply_anomaly(image, mask, anomaly_type, rng):
    if anomaly_type == "thickening":
        return apply_thickening(image, mask)
    if anomaly_type == "thinning":
        return apply_thinning(image, mask)
    if anomaly_type == "blur":
        return apply_blur(image, mask)
    if anomaly_type == "dropout":
        return apply_dropout(image, mask, rng)
    return image


# ============================================================
# RESTORE / VERIFY EXISTING BENCHMARK DATASET
# ============================================================

def verify_or_restore_existing_dataset():
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
    print(f"Validated {len(df)} samples.")
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

    print("=" * 70)
    print("Regenerating 1,000 Synthetic Form Anomalies Strictly Within Handwriting Region")
    print("=" * 70)

    rng = np.random.RandomState(RANDOM_SEED)

    image_map = find_images()
    bbox_map = load_bbox_map()
    print(f"Indexed {len(image_map)} IAM form images and {len(bbox_map)} handwriting bounding boxes.")

    paired_df = pd.read_csv(PAIRED_CSV)
    print(f"Generating aligned dataset matching {len(paired_df)} paired source forms...")

    records = []
    for idx, row in paired_df.iterrows():
        sample_id = str(row["sample_id"])
        source_form = str(row["source_form"])
        anomaly_type = str(row["anomaly_type"])

        if source_form not in image_map:
            raise RuntimeError(f"Source form {source_form} not found in IAM images!")

        image = cv2.imread(str(image_map[source_form]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read source image {image_map[source_form]}!")

        # Create mask strictly inside handwriting bounding box
        anomaly_mask = create_random_handwriting_mask(image, source_form, bbox_map, rng)
        attempts = 0
        while np.sum(anomaly_mask > 0) == 0 and attempts < 10:
            anomaly_mask = create_random_handwriting_mask(image, source_form, bbox_map, rng)
            attempts += 1

        if np.sum(anomaly_mask > 0) == 0:
            raise RuntimeError(f"Failed to generate valid handwriting mask for {source_form}!")

        synthetic_image = apply_anomaly(image, anomaly_mask, anomaly_type, rng)

        image_path = IMAGE_OUTPUT / f"{sample_id}.png"
        mask_path = MASK_OUTPUT / f"{sample_id}_mask.png"

        cv2.imwrite(str(image_path), synthetic_image)
        cv2.imwrite(str(mask_path), anomaly_mask)

        records.append({
            "sample_id": sample_id,
            "source_form": source_form,
            "synthetic_image": str(image_path.relative_to(PROJECT_ROOT)),
            "ground_truth_mask": str(mask_path.relative_to(PROJECT_ROOT)),
            "anomaly_type": anomaly_type,
            "label": 1,
            "mask_pixels": int(np.sum(anomaly_mask > 0)),
        })

        if (idx + 1) % 100 == 0:
            print(f"Generated {idx + 1}/{NUM_SAMPLES} samples...")

    df = pd.DataFrame(records)
    df.to_csv(CSV_OUTPUT, index=False)

    print()
    print("SUCCESS: 1,000 synthetic form anomalies regenerated strictly on handwriting!")
    print(f"Images: {IMAGE_OUTPUT}")
    print(f"Masks:  {MASK_OUTPUT}")
    print(f"CSV:    {CSV_OUTPUT}")
    print("\nAnomaly distribution:")
    print(df["anomaly_type"].value_counts())


if __name__ == "__main__":
    main()