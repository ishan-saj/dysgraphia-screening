from pathlib import Path
import pandas as pd
import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORDS_METADATA_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "words_metadata.csv"
)

IAM_IMAGES_DIR = (
    PROJECT_ROOT / "data" / "raw" / "IAM" / "images"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "segmentation" / "word_crops"
)


# ============================================================
# SETTINGS
# ============================================================

# Small padding around each word.
# Set to 0 if you want the exact bounding box only.
PADDING = 2

# Do not overwrite existing crops.
SKIP_EXISTING = True


# ============================================================
# FIND IAM FORM IMAGE
# ============================================================

def find_form_image(form_id):
    """
    Find the exact IAM form image.

    Example:
        l01-179 -> l01-179.png
    """

    filename = f"{form_id}.png"

    # Check known IAM directories first.
    possible_dirs = [
        IAM_IMAGES_DIR / "formsA-D",
        IAM_IMAGES_DIR / "formsE-H",
        IAM_IMAGES_DIR / "formsI-Z",
    ]

    for directory in possible_dirs:
        candidate = directory / filename

        if candidate.exists():
            return candidate

    # Fallback recursive search.
    matches = list(IAM_IMAGES_DIR.rglob(filename))

    if matches:
        return matches[0]

    return None


# ============================================================
# SAFE INTEGER
# ============================================================

def safe_int(value):
    try:
        return int(round(float(value)))
    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IAM WORD CROP GENERATION")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Metadata     : {WORDS_METADATA_FILE}")
    print(f"IAM images   : {IAM_IMAGES_DIR}")
    print(f"Output       : {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # Check required files/directories
    # --------------------------------------------------------

    if not WORDS_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"words_metadata.csv not found:\n{WORDS_METADATA_FILE}"
        )

    if not IAM_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"IAM image directory not found:\n{IAM_IMAGES_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print("Loading word metadata...")

    df = pd.read_csv(WORDS_METADATA_FILE)

    print(f"Total metadata rows: {len(df):,}")

    required_columns = [
        "word_id",
        "form_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Keep valid segmentation records
    # --------------------------------------------------------

    if "segmentation_status" in df.columns:

        before = len(df)

        df = df[
            df["segmentation_status"]
            .astype(str)
            .str.lower()
            .eq("ok")
        ].copy()

        print(
            f"Rows with segmentation_status='ok': "
            f"{len(df):,} / {before:,}"
        )

    # --------------------------------------------------------
    # Remove duplicate word IDs
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates(
        subset=["word_id"],
        keep="first"
    ).copy()

    print(
        f"Unique word IDs: {len(df):,} "
        f"(removed {before - len(df):,} duplicates)"
    )

    # --------------------------------------------------------
    # Form list
    # --------------------------------------------------------

    form_ids = (
        df["form_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(f"Unique forms in metadata: {len(form_ids):,}")
    print()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    forms_processed = 0
    forms_missing = 0

    crops_created = 0
    crops_skipped = 0
    crops_invalid = 0
    crops_failed = 0

    missing_forms = []
    invalid_examples = []

    # Cache form images so the same form isn't repeatedly read
    image_cache = {}

    # --------------------------------------------------------
    # Process each form
    # --------------------------------------------------------

    for form_index, form_id in enumerate(form_ids, start=1):

        form_id = str(form_id)

        form_rows = df[
            df["form_id"].astype(str) == form_id
        ]

        image_path = find_form_image(form_id)

        if image_path is None:

            forms_missing += 1
            missing_forms.append(form_id)

            print(
                f"[{form_index}/{len(form_ids)}] "
                f"MISSING FORM: {form_id}"
            )

            continue

        # ----------------------------------------------------
        # Read image once
        # ----------------------------------------------------

        if form_id not in image_cache:

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:

                forms_missing += 1
                missing_forms.append(form_id)

                print(
                    f"[{form_index}/{len(form_ids)}] "
                    f"FAILED TO READ: {form_id}"
                )

                continue

            image_cache[form_id] = image

        else:
            image = image_cache[form_id]

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # Create sample-specific directory
        # ----------------------------------------------------

        sample_output_dir = OUTPUT_DIR / form_id

        sample_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Process words
        # ----------------------------------------------------

        for _, row in form_rows.iterrows():

            word_id = str(row["word_id"])

            output_path = (
                sample_output_dir /
                f"{word_id}.png"
            )

            # ----------------------------------------------
            # Never overwrite existing crop
            # ----------------------------------------------

            if SKIP_EXISTING and output_path.exists():

                crops_skipped += 1
                continue

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            x = safe_int(row["bbox_x"])
            y = safe_int(row["bbox_y"])
            w = safe_int(row["bbox_width"])
            h = safe_int(row["bbox_height"])

            if (
                x is None
                or y is None
                or w is None
                or h is None
                or w <= 0
                or h <= 0
            ):

                crops_invalid += 1

                if len(invalid_examples) < 20:
                    invalid_examples.append(
                        (word_id, "invalid bounding box")
                    )

                continue

            # ----------------------------------------------
            # Apply padding
            # ----------------------------------------------

            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)

            x2 = min(
                image_width,
                x + w + PADDING
            )

            y2 = min(
                image_height,
                y + h + PADDING
            )

            if x1 >= x2 or y1 >= y2:

                crops_invalid += 1

                if len(invalid_examples) < 20:
                    invalid_examples.append(
                        (word_id, "box outside image")
                    )

                continue

            # ----------------------------------------------
            # Crop
            # ----------------------------------------------

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:

                crops_invalid += 1

                if len(invalid_examples) < 20:
                    invalid_examples.append(
                        (word_id, "empty crop")
                    )

                continue

            # ----------------------------------------------
            # Save
            # ----------------------------------------------

            success = cv2.imwrite(
                str(output_path),
                crop
            )

            if success:
                crops_created += 1
            else:
                crops_failed += 1

        forms_processed += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            form_index == 1
            or form_index % 50 == 0
            or form_index == len(form_ids)
        ):

            print(
                f"[{form_index}/{len(form_ids)}] "
                f"{form_id} | "
                f"created={crops_created:,} | "
                f"skipped={crops_skipped:,}"
            )

    # --------------------------------------------------------
    # Clear cache
    # --------------------------------------------------------

    image_cache.clear()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("WORD CROP GENERATION COMPLETE")
    print("=" * 70)

    print(f"Forms processed       : {forms_processed:,}")
    print(f"Forms missing         : {forms_missing:,}")
    print(f"Word crops created    : {crops_created:,}")
    print(f"Existing crops skipped: {crops_skipped:,}")
    print(f"Invalid boxes         : {crops_invalid:,}")
    print(f"Save failures         : {crops_failed:,}")

    print()
    print(f"Output directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Missing forms
    # --------------------------------------------------------

    if missing_forms:

        print()
        print("Missing forms:")

        for form_id in missing_forms[:50]:
            print(f"  {form_id}")

        if len(missing_forms) > 50:
            print(
                f"  ... and "
                f"{len(missing_forms) - 50} more"
            )

    # --------------------------------------------------------
    # Invalid examples
    # --------------------------------------------------------

    if invalid_examples:

        print()
        print("Invalid crop examples:")

        for word_id, reason in invalid_examples:
            print(
                f"  {word_id}: {reason}"
            )

    print()
    print("Example expected output:")
    print(
        OUTPUT_DIR /
        "l01-179" /
        "l01-179-04-01.png"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()