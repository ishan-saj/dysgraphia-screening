import csv
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "IAM" / "ascii" / "words.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "metadata" / "words_metadata.csv"


# ============================================================
# PARSE ONE WORD
# ============================================================

def parse_word(line):
    """
    Parse one data line from IAM words.txt.

    Expected format based on the actual data:

    word_id status graylevel x y width height tag transcription

    Example:
    a01-000u-02-08 ok 157 2092 1121 302 65 NN Labour
    """

    parts = line.strip().split()

    # Need at least:
    # word_id
    # status
    # graylevel
    # x
    # y
    # width
    # height
    # tag
    # transcription
    if len(parts) < 9:
        print("Warning: could not parse line:")
        print(line)
        return None

    try:
        # ----------------------------------------------------
        # Basic word information
        # ----------------------------------------------------

        word_id = parts[0]
        segmentation_status = parts[1]
        graylevel = int(parts[2])

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        bbox_x = int(parts[3])
        bbox_y = int(parts[4])
        bbox_width = int(parts[5])
        bbox_height = int(parts[6])

        # ----------------------------------------------------
        # Linguistic information
        # ----------------------------------------------------

        grammatical_tag = parts[7]

        # Transcription is everything after the grammatical tag.
        # This allows transcriptions containing spaces.
        transcription = " ".join(parts[8:])

        # ----------------------------------------------------
        # Extract form ID and line ID
        #
        # Example:
        #
        # a01-000u-02-08
        #
        # form_id = a01-000u
        # line_id = a01-000u-02
        # ----------------------------------------------------

        word_parts = word_id.split("-")

        if len(word_parts) < 4:
            print("Warning: unexpected word ID:")
            print(line)
            return None

        form_id = "-".join(word_parts[:2])
        line_id = "-".join(word_parts[:3])

        # ----------------------------------------------------
        # Return structured row
        # ----------------------------------------------------

        return {
            "word_id": word_id,
            "form_id": form_id,
            "line_id": line_id,
            "segmentation_status": segmentation_status,
            "graylevel": graylevel,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "grammatical_tag": grammatical_tag,
            "transcription": transcription
        }

    except ValueError:
        print("Warning: could not parse line:")
        print(line)
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        print("ERROR: words.txt was not found.")
        print(f"Expected location: {INPUT_FILE}")
        return

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    total_lines = 0
    skipped_lines = 0

    # --------------------------------------------------------
    # Read words.txt
    # --------------------------------------------------------

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            total_lines += 1

            line = line.strip()

            # ------------------------------------------------
            # Skip empty lines
            # ------------------------------------------------

            if not line:
                skipped_lines += 1
                continue

            # ------------------------------------------------
            # Skip comments/header information
            #
            # Example:
            # #--- words.txt ---
            # # iam database word information
            # ------------------------------------------------

            if line.startswith("#"):
                skipped_lines += 1
                continue

            # ------------------------------------------------
            # Parse word
            # ------------------------------------------------

            row = parse_word(line)

            if row is not None:
                rows.append(row)
            else:
                skipped_lines += 1

    # --------------------------------------------------------
    # CSV columns
    # --------------------------------------------------------

    fieldnames = [
        "word_id",
        "form_id",
        "line_id",
        "segmentation_status",
        "graylevel",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "grammatical_tag",
        "transcription"
    ]

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("Word metadata extraction complete.")
    print("----------------------------------")
    print(f"Input file:       {INPUT_FILE}")
    print(f"Output file:      {OUTPUT_FILE}")
    print(f"Total lines read: {total_lines}")
    print(f"Words extracted:  {len(rows)}")
    print(f"Lines skipped:    {skipped_lines}")
    print("----------------------------------")
    print("CSV columns:")
    
    for column in fieldnames:
        print(f"  - {column}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()