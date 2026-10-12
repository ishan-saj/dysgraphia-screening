#!/usr/bin/env python3

"""
Extract all line-level metadata from the IAM Handwriting Database lines.txt.

Input:
    data/raw/IAM/ascii/lines.txt

Output:
    data/metadata/lines_metadata.csv

One row = one IAM handwriting line.
"""

from pathlib import Path
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

IAM_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM"

LINES_TXT = IAM_ROOT / "ascii" / "lines.txt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_CSV = OUTPUT_DIR / "lines_metadata.csv"


# ============================================================
# PARSE ONE LINE
# ============================================================

def parse_line(line):
    """
    Parse one data line from IAM lines.txt.

    IAM format:

    line_id segmentation_status graylevel num_components
    x y width height transcription

    Example:

    a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from
    """

    line = line.strip()

    # Ignore empty lines
    if not line:
        return None

    # Ignore comments
    if line.startswith("#"):
        return None

    # Split only the first 9 whitespace-separated fields.
    #
    # The final field is transcription and is allowed to contain
    # the | character separating words.
    parts = line.split(maxsplit=9)

    # We need exactly:
    #
    # 0 line_id
    # 1 segmentation_status
    # 2 graylevel
    # 3 num_components
    # 4 bbox_x
    # 5 bbox_y
    # 6 bbox_width
    # 7 bbox_height
    # 8 transcription
    #
    if len(parts) != 9:
        print(
            f"Warning: could not parse line:\n{line}"
        )
        return None

    line_id = parts[0]

    segmentation_status = parts[1]

    graylevel = int(parts[2])

    num_components = int(parts[3])

    bbox_x = int(parts[4])

    bbox_y = int(parts[5])

    bbox_width = int(parts[6])

    bbox_height = int(parts[7])

    transcription = parts[8]

    # --------------------------------------------------------
    # Extract form_id from line_id
    #
    # Example:
    # a01-000u-00
    #       ↓
    # a01-000u
    # --------------------------------------------------------

    form_id = "-".join(
        line_id.split("-")[:-1]
    )

    # --------------------------------------------------------
    # Number of words
    #
    # IAM separates word tokens using |
    # --------------------------------------------------------

    if transcription:
        num_words = len(
            transcription.split("|")
        )
    else:
        num_words = 0

    return {
        "line_id": line_id,
        "form_id": form_id,
        "segmentation_status": segmentation_status,
        "graylevel": graylevel,
        "num_components": num_components,
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "transcription": transcription,
        "num_words": num_words,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not LINES_TXT.exists():

        raise FileNotFoundError(
            f"lines.txt not found:\n{LINES_TXT}"
        )

    # --------------------------------------------------------
    # Read and parse lines.txt
    # --------------------------------------------------------

    rows = []

    with LINES_TXT.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            row = parse_line(line)

            if row is not None:
                rows.append(row)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CSV columns
    # --------------------------------------------------------

    fieldnames = [
        "line_id",
        "form_id",
        "segmentation_status",
        "graylevel",
        "num_components",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "transcription",
        "num_words",
    ]

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("IAM LINE METADATA EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Input:  {LINES_TXT}"
    )

    print(
        f"Output: {OUTPUT_CSV}"
    )

    print(
        f"Lines extracted: {len(rows)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()