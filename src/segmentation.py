from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

SAMPLE_ID = "p06-052"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "IAM"
    / "images"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "segmentation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(sample_id):

    matches = list(IMAGE_ROOT.rglob(f"{sample_id}.png"))

    if not matches:
        matches = list(IMAGE_ROOT.rglob(f"{sample_id}.jpg"))

    if not matches:
        raise FileNotFoundError(
            f"Image not found: {sample_id}"
        )

    return matches[0]


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return image


# ============================================================
# INITIAL INK MASK
# ============================================================

def create_ink_mask(gray):

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        15
    )

    return mask


# ============================================================
# REMOVE HORIZONTAL RULES
# ============================================================

def remove_horizontal_lines(mask):

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (100, 1)
    )

    horizontal_lines = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    cleaned = cv2.subtract(
        mask,
        horizontal_lines
    )

    return cleaned


# ============================================================
# REMOVE SMALL NOISE
# ============================================================

def remove_noise(mask):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Remove tiny isolated noise
        if area < 20:
            continue

        # Remove very thin vertical border artifacts
        if h > 500 and w < 20:
            continue

        cleaned[labels == i] = 255

    return cleaned


# ============================================================
# FIND HANDWRITING REGION
# ============================================================

def find_handwriting_region(mask):

    height, width = mask.shape

    horizontal_projection = (
        mask > 0
    ).sum(axis=1)

    # Smooth projection
    kernel_size = 31

    kernel = np.ones(
        kernel_size,
        dtype=np.float32
    ) / kernel_size

    smooth = np.convolve(
        horizontal_projection,
        kernel,
        mode="same"
    )

    # Ignore the upper part of the page where
    # printed header/reference text exists.
    #
    # IAM sentence forms normally contain the
    # handwritten response in the lower portion.

    search_start = int(height * 0.30)

    search_region = smooth[search_start:]

    if len(search_region) == 0:
        return search_start, height

    # Find rows containing substantial ink
    threshold = max(
        15,
        np.percentile(
            search_region,
            35
        )
    )

    active = (
        search_region > threshold
    )

    # Find first substantial handwriting area
    runs = []

    start = None

    for i, value in enumerate(active):

        if value and start is None:
            start = i

        elif not value and start is not None:

            end = i

            if end - start > 20:
                runs.append(
                    (start, end)
                )

            start = None

    if start is not None:
        runs.append(
            (start, len(active))
        )

    if not runs:
        return search_start, height

    # The handwriting area is normally the
    # longest/most substantial region.
    best_run = max(
        runs,
        key=lambda x: x[1] - x[0]
    )

    y1 = search_start + best_run[0]

    # Add some margin
    y1 = max(
        0,
        y1 - 30
    )

    y2 = height

    return y1, y2


# ============================================================
# DETECT HANDWRITING LINES
# ============================================================

def detect_lines(mask, y1, y2):

    region = mask[y1:y2, :]

    projection = (
        region > 0
    ).sum(axis=1)

    # Smooth horizontal projection
    kernel = np.ones(
        25,
        dtype=np.float32
    ) / 25

    smooth = np.convolve(
        projection,
        kernel,
        mode="same"
    )

    # Ignore extremely weak rows
    threshold = max(
        3,
        int(mask.shape[1] * 0.005)
    )

    active = (
        smooth > threshold
    )

    # Close small gaps inside handwriting lines
    active_uint8 = (
        active.astype(np.uint8) * 255
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 12)
    )

    active_uint8 = cv2.morphologyEx(
        active_uint8.reshape(-1, 1),
        cv2.MORPH_CLOSE,
        close_kernel
    ).flatten()

    active = (
        active_uint8 > 0
    )

    lines = []

    start = None

    for i, value in enumerate(active):

        if value and start is None:

            start = i

        elif not value and start is not None:

            end = i

            if end - start >= 15:

                lines.append(
                    (
                        y1 + start,
                        y1 + end
                    )
                )

            start = None

    if start is not None:

        lines.append(
            (
                y1 + start,
                y2
            )
        )

    # Remove giant regions caused by merged lines
    cleaned_lines = []

    for line_start, line_end in lines:

        height = line_end - line_start

        # Normal handwriting line height
        if height < mask.shape[0] * 0.15:

            cleaned_lines.append(
                (
                    line_start,
                    line_end
                )
            )

    return cleaned_lines


# ============================================================
# WORD DETECTION
# ============================================================

def detect_words(mask, line):

    y1, y2 = line

    line_mask = mask[
        y1:y2,
        :
    ]

    # --------------------------------------------------------
    # Dilate horizontally.
    #
    # This is VERY important.
    #
    # It connects letters belonging to the same word,
    # while preserving larger gaps between words.
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 5)
    )

    word_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Slight dilation
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 3)
    )

    word_mask = cv2.dilate(
        word_mask,
        dilation_kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        word_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    line_height = y2 - y1

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = w * h

        # Ignore tiny punctuation/noise
        if area < 200:
            continue

        if w < 20:
            continue

        if h < max(10, line_height * 0.15):
            continue

        # Ignore extremely large accidental regions
        if w > mask.shape[1] * 0.50:
            continue

        candidates.append(
            (
                x,
                y1 + y,
                w,
                h
            )
        )

    # Sort left → right
    candidates.sort(
        key=lambda box: box[0]
    )

    # --------------------------------------------------------
    # Merge boxes that are very close horizontally.
    # --------------------------------------------------------

    words = []

    for box in candidates:

        if not words:

            words.append(box)
            continue

        px, py, pw, ph = words[-1]

        x, y, w, h = box

        previous_right = px + pw

        gap = x - previous_right

        # If two regions are very close,
        # treat them as one word.
        max_gap = max(
            15,
            int(line_height * 0.12)
        )

        if gap < max_gap:

            new_x = min(px, x)
            new_y = min(py, y)

            new_right = max(
                px + pw,
                x + w
            )

            new_bottom = max(
                py + ph,
                y + h
            )

            words[-1] = (
                new_x,
                new_y,
                new_right - new_x,
                new_bottom - new_y
            )

        else:

            words.append(box)

    return words


# ============================================================
# DRAW VISUALIZATION
# ============================================================

def create_visualization(
    original,
    clean_mask,
    lines,
    all_words
):

    # Clean handwriting image
    clean = np.full_like(
        original,
        255
    )

    clean[clean_mask > 0] = 0

    clean_rgb = cv2.cvtColor(
        clean,
        cv2.COLOR_GRAY2BGR
    )

    # Draw boxes
    visualization = clean_rgb.copy()

    word_number = 1

    for line_number, words in enumerate(all_words):

        if line_number >= len(lines):
            break

        ly1, ly2 = lines[line_number]

        # Blue = line
        cv2.rectangle(
            visualization,
            (0, ly1),
            (
                original.shape[1] - 1,
                ly2
            ),
            (255, 0, 0),
            3
        )

        cv2.putText(
            visualization,
            f"LINE {line_number + 1}",
            (10, max(20, ly1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        for x, y, w, h in words:

            # Red = word
            cv2.rectangle(
                visualization,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                visualization,
                str(word_number),
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            word_number += 1

    return clean, visualization


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HANDWRITING-FIRST SEGMENTATION")
    print("=" * 60)

    image_path = find_image(
        SAMPLE_ID
    )

    print("\nImage:")
    print(image_path)

    gray = load_image(
        image_path
    )

    print(
        f"\nImage size: "
        f"{gray.shape[1]} x {gray.shape[0]}"
    )

    # --------------------------------------------------------
    # 1. Ink
    # --------------------------------------------------------

    print("\n[1/6] Creating ink mask...")

    mask = create_ink_mask(
        gray
    )

    # --------------------------------------------------------
    # 2. Remove horizontal rules
    # --------------------------------------------------------

    print(
        "[2/6] Removing horizontal rules..."
    )

    mask = remove_horizontal_lines(
        mask
    )

    # --------------------------------------------------------
    # 3. Remove noise
    # --------------------------------------------------------

    print(
        "[3/6] Removing noise..."
    )

    mask = remove_noise(
        mask
    )

    # --------------------------------------------------------
    # 4. Find handwriting region
    # --------------------------------------------------------

    print(
        "[4/6] Finding handwriting region..."
    )

    handwriting_y1, handwriting_y2 = (
        find_handwriting_region(
            mask
        )
    )

    print(
        f"Handwriting region: "
        f"y={handwriting_y1} "
        f"to y={handwriting_y2}"
    )

    handwriting_mask = np.zeros_like(
        mask
    )

    handwriting_mask[
        handwriting_y1:handwriting_y2,
        :
    ] = mask[
        handwriting_y1:handwriting_y2,
        :
    ]

    # --------------------------------------------------------
    # 5. Lines
    # --------------------------------------------------------

    print(
        "[5/6] Detecting handwriting lines..."
    )

    lines = detect_lines(
        handwriting_mask,
        handwriting_y1,
        handwriting_y2
    )

    print(
        f"Detected handwriting lines: "
        f"{len(lines)}"
    )

    # --------------------------------------------------------
    # 6. Words
    # --------------------------------------------------------

    print(
        "[6/6] Detecting handwriting words..."
    )

    all_words = []

    total_words = 0

    for i, line in enumerate(lines):

        words = detect_words(
            handwriting_mask,
            line
        )

        all_words.append(
            words
        )

        total_words += len(words)

        print(
            f"Line {i + 1}: "
            f"{len(words)} word regions"
        )

    print(
        f"\nTotal detected word regions: "
        f"{total_words}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    clean, visualization = (
        create_visualization(
            gray,
            handwriting_mask,
            lines,
            all_words
        )
    )

    mask_path = (
        OUTPUT_DIR
        / f"{SAMPLE_ID}_ink_mask.png"
    )

    clean_path = (
        OUTPUT_DIR
        / f"{SAMPLE_ID}_clean.png"
    )

    segmentation_path = (
        OUTPUT_DIR
        / f"{SAMPLE_ID}_segmentation.png"
    )

    cv2.imwrite(
        str(mask_path),
        handwriting_mask
    )

    cv2.imwrite(
        str(clean_path),
        clean
    )

    cv2.imwrite(
        str(segmentation_path),
        visualization
    )

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)

    print(
        f"\nInk mask:\n{mask_path}"
    )

    print(
        f"\nClean handwriting:\n{clean_path}"
    )

    print(
        f"\nSegmentation:\n{segmentation_path}"
    )


if __name__ == "__main__":
    main()