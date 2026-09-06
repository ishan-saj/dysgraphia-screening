import csv
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "IAM" / "ascii" / "sentences.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "metadata" / "sentences_metadata.csv"


# ============================================================
# PARSE ONE SENTENCE
# ============================================================

def parse_sentence(line):
    """
    Parse one data line from IAM sentences.txt.

    Format:

    sentence_id sentence_number status graylevel num_components
    x y width height transcription

    Example:

    a01-000u-s00-00 0 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from
    """

    # Split only the first 9 spaces.
    # This keeps the entire transcription together.
    parts = line.strip().split(maxsplit=9)

    # Expected:
    #
    # 0 = sentence_id
    # 1 = sentence_number
    # 2 = segmentation_status
    # 3 = graylevel
    # 4 = num_components
    # 5 = bbox_x
    # 6 = bbox_y
    # 7 = bbox_width
    # 8 = bbox_height
    # 9 = transcription
    #
    if len(parts) != 10:
        print("Warning: could not parse line:")
        print(line)
        return None

    try:

        # ----------------------------------------------------
        # Basic sentence information
        # ----------------------------------------------------

        sentence_id = parts[0]
        sentence_number = int(parts[1])
        segmentation_status = parts[2]
        graylevel = int(parts[3])
        num_components = int(parts[4])

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        bbox_x = int(parts[5])
        bbox_y = int(parts[6])
        bbox_width = int(parts[7])
        bbox_height = int(parts[8])

        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        transcription = parts[9]

        # ----------------------------------------------------
        # Extract form ID
        #
        # Example:
        #
        # a01-000u-s00-00
        #
        # form_id = a01-000u
        # ----------------------------------------------------

        sentence_parts = sentence_id.split("-")

        if len(sentence_parts) < 4:
            print("Warning: unexpected sentence ID:")
            print(line)
            return None

        form_id = "-".join(sentence_parts[:2])

        # ----------------------------------------------------
        # Count words
        #
        # Words in the transcription are separated by |
        # ----------------------------------------------------

        if transcription:
            num_words = len(transcription.split("|"))
        else:
            num_words = 0

        # ----------------------------------------------------
        # Return structured row
        # ----------------------------------------------------

        return {
            "sentence_id": sentence_id,
            "form_id": form_id,
            "sentence_number": sentence_number,
            "segmentation_status": segmentation_status,
            "graylevel": graylevel,
            "num_components": num_components,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "transcription": transcription,
            "num_words": num_words
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
        print("ERROR: sentences.txt was not found.")
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
    # Read sentences.txt
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
            # ------------------------------------------------

            if line.startswith("#"):
                skipped_lines += 1
                continue

            # ------------------------------------------------
            # Parse sentence
            # ------------------------------------------------

            row = parse_sentence(line)

            if row is not None:
                rows.append(row)
            else:
                skipped_lines += 1

    # --------------------------------------------------------
    # CSV columns
    # --------------------------------------------------------

    fieldnames = [
        "sentence_id",
        "form_id",
        "sentence_number",
        "segmentation_status",
        "graylevel",
        "num_components",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "transcription",
        "num_words"
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
    print("Sentence metadata extraction complete.")
    print("---------------------------------------")
    print(f"Input file:        {INPUT_FILE}")
    print(f"Output file:       {OUTPUT_FILE}")
    print(f"Total lines read:  {total_lines}")
    print(f"Sentences extracted: {len(rows)}")
    print(f"Lines skipped:      {skipped_lines}")
    print("---------------------------------------")
    print("CSV columns:")

    for column in fieldnames:
        print(f"  - {column}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()