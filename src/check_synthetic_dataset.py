from pathlib import Path
import cv2
import pandas as pd
import numpy as np


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORD_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_anomalies"
    / "synthetic_anomalies.csv"
)

FORM_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_form_anomalies.csv"
)

IAM_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IAM"
)


# ==================================================
# PATH RESOLVER
# ==================================================

def resolve_path(path_value):

    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ==================================================
# FIND IAM FORM
# ==================================================

def find_iam_form(source_form):

    source_form = str(source_form)

    # Search for the IAM form image
    matches = list(
        IAM_ROOT.rglob(source_form + ".png")
    )

    if matches:
        return matches[0]

    # Some datasets may have the extension already
    matches = list(
        IAM_ROOT.rglob(source_form)
    )

    if matches:
        return matches[0]

    return None


# ==================================================
# VALIDATE DATASET
# ==================================================

def validate_dataset(
    csv_path,
    dataset_name,
    original_column,
    form_level=False
):

    print()
    print("=" * 55)
    print(f"{dataset_name} Validation")
    print("=" * 55)

    # --------------------------------------------------
    # CSV
    # --------------------------------------------------

    if not csv_path.exists():

        print("CSV not found:")
        print(csv_path)

        return False

    df = pd.read_csv(csv_path)

    print(f"CSV samples: {len(df)}")

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    required_columns = [
        "sample_id",
        original_column,
        "synthetic_image",
        "ground_truth_mask",
        "anomaly_type"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        print()
        print("Missing columns:")

        for col in missing_columns:
            print(f"  - {col}")

        return False

    # --------------------------------------------------
    # Anomaly types
    # --------------------------------------------------

    print()
    print("Anomaly types:")
    print(df["anomaly_type"].value_counts())

    # --------------------------------------------------
    # Counters
    # --------------------------------------------------

    valid_samples = 0
    invalid_samples = 0

    mask_pixels = []
    changed_pixels = []

    # ==================================================
    # CHECK EACH SAMPLE
    # ==================================================

    for _, row in df.iterrows():

        # --------------------------------------------------
        # ORIGINAL
        # --------------------------------------------------

        if form_level:

            # source_form contains IAM form ID
            original_path = find_iam_form(
                row[original_column]
            )

            if original_path is None:

                print(
                    f"Could not find IAM form: "
                    f"{row['sample_id']} "
                    f"({row[original_column]})"
                )

                invalid_samples += 1
                continue

        else:

            original_path = resolve_path(
                row[original_column]
            )

        # --------------------------------------------------
        # SYNTHETIC
        # --------------------------------------------------

        synthetic_path = resolve_path(
            row["synthetic_image"]
        )

        # --------------------------------------------------
        # MASK
        # --------------------------------------------------

        mask_path = resolve_path(
            row["ground_truth_mask"]
        )

        # --------------------------------------------------
        # LOAD
        # --------------------------------------------------

        original = cv2.imread(
            str(original_path),
            cv2.IMREAD_GRAYSCALE
        )

        synthetic = cv2.imread(
            str(synthetic_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        # --------------------------------------------------
        # READ CHECKS
        # --------------------------------------------------

        if original is None:

            print(
                f"Could not read original: "
                f"{row['sample_id']}"
            )

            invalid_samples += 1
            continue

        if synthetic is None:

            print(
                f"Could not read synthetic: "
                f"{row['sample_id']}"
            )

            invalid_samples += 1
            continue

        if mask is None:

            print(
                f"Could not read mask: "
                f"{row['sample_id']}"
            )

            invalid_samples += 1
            continue

        # --------------------------------------------------
        # DIMENSIONS
        # --------------------------------------------------

        if (
            original.shape != synthetic.shape
            or original.shape != mask.shape
        ):

            print(
                f"Dimension mismatch: "
                f"{row['sample_id']}"
            )

            invalid_samples += 1
            continue

        # --------------------------------------------------
        # MASK PIXELS
        # --------------------------------------------------

        pixels = np.sum(mask > 0)

        if pixels == 0:

            print(
                f"Empty mask: "
                f"{row['sample_id']}"
            )

            invalid_samples += 1
            continue

        mask_pixels.append(pixels)

        # --------------------------------------------------
        # IMAGE DIFFERENCE
        # --------------------------------------------------

        difference = cv2.absdiff(
            original,
            synthetic
        )

        changed = np.sum(difference > 0)

        if changed == 0:

            print(
                f"No image change: "
                f"{row['sample_id']}"
            )

            invalid_samples += 1
            continue

        changed_pixels.append(changed)

        valid_samples += 1

    # ==================================================
    # RESULTS
    # ==================================================

    print()
    print("-" * 55)
    print("Validation Results")
    print("-" * 55)

    print(
        f"Valid samples   : {valid_samples}"
    )

    print(
        f"Invalid samples : {invalid_samples}"
    )

    if mask_pixels:

        print(
            f"Minimum mask pixels : "
            f"{min(mask_pixels)}"
        )

        print(
            f"Maximum mask pixels : "
            f"{max(mask_pixels)}"
        )

        print(
            f"Average mask pixels : "
            f"{np.mean(mask_pixels):.1f}"
        )

    if changed_pixels:

        print(
            f"Minimum changed pixels : "
            f"{min(changed_pixels)}"
        )

        print(
            f"Maximum changed pixels : "
            f"{max(changed_pixels)}"
        )

        print(
            f"Average changed pixels : "
            f"{np.mean(changed_pixels):.1f}"
        )

    print()

    if invalid_samples == 0:

        print(f"{dataset_name}: PASS")
        print("All synthetic samples are valid.")

        return True

    else:

        print(f"{dataset_name}: WARNING")
        print("Some samples need investigation.")

        return False


# ==================================================
# WORD LEVEL
# ==================================================

word_pass = validate_dataset(
    WORD_CSV_PATH,
    "WORD-LEVEL Synthetic Dataset",
    "source_word",
    form_level=False
)


# ==================================================
# FORM LEVEL
# ==================================================

form_pass = validate_dataset(
    FORM_CSV_PATH,
    "FORM-LEVEL Synthetic Dataset",
    "source_form",
    form_level=True
)


# ==================================================
# FINAL
# ==================================================

print()
print("=" * 55)
print("FINAL SYNTHETIC DATASET STATUS")
print("=" * 55)

print(
    f"Word-level : "
    f"{'PASS' if word_pass else 'FAIL'}"
)

print(
    f"Form-level : "
    f"{'PASS' if form_pass else 'FAIL'}"
)

print()

if word_pass and form_pass:

    print("OVERALL STATUS: PASS")
    print("Both synthetic datasets are valid.")

else:

    print("OVERALL STATUS: WARNING")
    print("At least one dataset needs investigation.")