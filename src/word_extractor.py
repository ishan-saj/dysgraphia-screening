from pathlib import Path
import re

import cv2
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FORM_ID = "p06-052"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM" / "images"
WORDS_METADATA = PROJECT_ROOT / "data" / "metadata" / "words_metadata.csv"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "metadata" / "word_crops"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FORM_OUTPUT = OUTPUT_ROOT / FORM_ID
FORM_OUTPUT.mkdir(parents=True, exist_ok=True)

VISUALIZATION_PATH = (
    PROJECT_ROOT / "data" / "metadata" / f"{FORM_ID}_iam_word_boxes.png"
)

FILTERED_METADATA_PATH = (
    PROJECT_ROOT / "data" / "metadata" / f"{FORM_ID}_word_metadata.csv"
)

PADDING = 10


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(form_id):
    matches = list(IMAGE_ROOT.rglob(f"{form_id}.png"))

    if not matches:
        matches = list(IMAGE_ROOT.rglob(f"{form_id}.jpg"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find image for {form_id} inside {IMAGE_ROOT}"
        )

    return matches[0]


# ============================================================
# CHECK PUNCTUATION
# ============================================================

def is_punctuation(text):
    """
    Returns True when the transcription contains only punctuation.
    These regions are extracted but excluded from word-level
    anomaly analysis later.
    """

    if pd.isna(text):
        return False

    text = str(text).strip()

    if not text:
        return True

    return bool(re.fullmatch(r"[^\w]+", text))


# ============================================================
# EXTRACT WORD CROP
# ============================================================

def extract_word(image, row):
    """
    Extract one word using the exact IAM bounding box.
    """

    image_height, image_width = image.shape[:2]

    x = int(row["bbox_x"])
    y = int(row["bbox_y"])
    w = int(row["bbox_width"])
    h = int(row["bbox_height"])

    # Add padding while staying inside image
    x1 = max(0, x - PADDING)
    y1 = max(0, y - PADDING)

    x2 = min(image_width, x + w + PADDING)
    y2 = min(image_height, y + h + PADDING)

    crop = image[y1:y2, x1:x2]

    return crop


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IAM WORD EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path = find_image(FORM_ID)

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    image_height, image_width = image.shape

    print(f"\nImage:")
    print(image_path)

    print(f"\nImage size: {image_width} x {image_height}")

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    if not WORDS_METADATA.exists():
        raise FileNotFoundError(
            f"Could not find metadata file:\n{WORDS_METADATA}"
        )

    words_df = pd.read_csv(WORDS_METADATA)

    required_columns = [
        "word_id",
        "form_id",
        "line_id",
        "segmentation_status",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "transcription",
    ]

    missing = [
        column for column in required_columns
        if column not in words_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in words_metadata.csv: {missing}"
        )

    # --------------------------------------------------------
    # Select current form
    # --------------------------------------------------------

    form_df = words_df[
        (words_df["form_id"] == FORM_ID)
        &
        (words_df["segmentation_status"].astype(str).str.lower() == "ok")
    ].copy()

    # Sort exactly by line and horizontal position
    form_df = form_df.sort_values(
        by=["line_id", "bbox_x"]
    ).reset_index(drop=True)

    print(f"\nIAM word regions found: {len(form_df)}")

    # --------------------------------------------------------
    # Prepare visualization
    # --------------------------------------------------------

    visualization = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    extracted_rows = []

    print("\nExtracting words...\n")

    # --------------------------------------------------------
    # Extract every IAM word
    # --------------------------------------------------------

    for index, row in form_df.iterrows():

        word_id = str(row["word_id"])
        line_id = str(row["line_id"])
        transcription = str(row["transcription"])

        crop = extract_word(image, row)

        if crop.size == 0:
            print(
                f"WARNING: Empty crop for {word_id}"
            )
            continue

        # Safe filename
        safe_word_id = word_id.replace("/", "_")

        crop_path = FORM_OUTPUT / f"{safe_word_id}.png"

        cv2.imwrite(
            str(crop_path),
            crop
        )

        # Original IAM bounding box
        x = int(row["bbox_x"])
        y = int(row["bbox_y"])
        w = int(row["bbox_width"])
        h = int(row["bbox_height"])

        # Draw rectangle
        cv2.rectangle(
            visualization,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

        # Draw line ID
        label = line_id.split("-")[-1]

        cv2.putText(
            visualization,
            f"L{label}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

        punctuation = is_punctuation(transcription)

        extracted_rows.append({
            "word_id": word_id,
            "form_id": FORM_ID,
            "line_id": line_id,
            "transcription": transcription,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": w,
            "bbox_height": h,
            "is_punctuation": punctuation,
            "crop_path": str(crop_path),
        })

        print(
            f"{len(extracted_rows):03d} | "
            f"{transcription:<20} | "
            f"line={line_id} | "
            f"{'PUNCTUATION' if punctuation else 'WORD'}"
        )

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    cv2.imwrite(
        str(VISUALIZATION_PATH),
        visualization
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    output_df = pd.DataFrame(extracted_rows)

    output_df.to_csv(
        FILTERED_METADATA_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    word_count = int(
        (~output_df["is_punctuation"]).sum()
    )

    punctuation_count = int(
        output_df["is_punctuation"].sum()
    )

    line_count = output_df["line_id"].nunique()

    print("\n" + "=" * 60)
    print("WORD EXTRACTION COMPLETED")
    print("=" * 60)

    print(f"Lines represented: {line_count}")
    print(f"Total IAM regions: {len(output_df)}")
    print(f"Actual word regions: {word_count}")
    print(f"Punctuation regions: {punctuation_count}")

    print(f"\nWord crops:")
    print(FORM_OUTPUT)

    print(f"\nVisualization:")
    print(VISUALIZATION_PATH)

    print(f"\nMetadata:")
    print(FILTERED_METADATA_PATH)


if __name__ == "__main__":
    main()

