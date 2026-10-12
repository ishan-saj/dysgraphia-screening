#!/usr/bin/env python3
"""
Create a 10-sample experimental metadata CSV from the IAM Handwriting Database.

Expected project structure:

dysgraphia-screening/
├── data/
│   ├── raw/
│   │   └── IAM/
│   │       ├── images/
│       │       ├── xml/
│       │       └── ascii/
│   └── metadata/
├── extract_metadata_sample.py
└── ...

Output:
data/metadata/metadata_sample_10.csv

IMPORTANT:
IAM does not provide dysgraphia diagnosis, age, or grade labels.

The 9 features extracted here are computational handwriting/image features.
They are NOT dysgraphia ground-truth labels.

Features:
1. Baseline Deviation
2. Word Spacing CV
3. Character Height CV
4. Average Word Height
5. Average Word Width
6. Stroke Density
7. Slant Angle
8. Writing Area
9. Connected Component Count
"""

from pathlib import Path
import csv
import math
import xml.etree.ElementTree as ET


# ============================================================
# PROJECT PATHS
# ============================================================

# The script is located in the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

IAM_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM"

XML_DIR = IAM_ROOT / "xml"
IMAGE_DIR = IAM_ROOT / "images"

OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_CSV = OUTPUT_DIR / "metadata_all.csv"




# ============================================================
# BASIC STATISTICS
# ============================================================

def mean(values):
    return sum(values) / len(values) if values else None


def std(values):
    if len(values) < 2:
        return 0.0 if values else None

    m = mean(values)

    return math.sqrt(
        sum((x - m) ** 2 for x in values) / len(values)
    )


def cv(values):
    """
    Coefficient of variation:

        CV = standard deviation / mean

    Used for Character Height CV and Word Spacing CV.
    """

    values = [
        v for v in values
        if v is not None and v > 0
    ]

    if not values:
        return None

    m = mean(values)

    if m == 0:
        return None

    return std(values) / m


def median(values):
    values = sorted(values)

    n = len(values)

    if n == 0:
        return None

    middle = n // 2

    if n % 2 == 1:
        return values[middle]

    return (
        values[middle - 1] + values[middle]
    ) / 2.0


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(form_id):
    """
    Find the PNG corresponding to an IAM form ID.

    Searches:
        formsA-D/
        formsE-H/
        formsI-Z/
    """

    matches = list(
        IMAGE_DIR.glob(
            f"*/{form_id}.png"
        )
    )

    return matches[0] if matches else None


# ============================================================
# SLANT ESTIMATION
# ============================================================

def estimate_line_slant(
    binary_crop,
    angle_range=(-45.0, 45.0),
    angle_step=2.0
):
    """
    Estimate line slant using projection-profile variance.

    Each candidate angle shears the line image.
    The angle producing the highest variance in the
    vertical projection profile is selected.

    This replaces whole-image PCA because PCA over the
    entire ink mass is strongly influenced by the horizontal
    writing direction.
    """

    import cv2
    import numpy as np

    h, w = binary_crop.shape

    if (
        h < 4
        or w < 4
        or binary_crop.sum() == 0
    ):
        return None

    best_angle = 0.0
    best_variance = -1.0

    angle = angle_range[0]

    while angle <= angle_range[1] + 1e-9:

        shear = math.tan(
            math.radians(angle)
        )

        pad = int(
            abs(shear) * h
        ) + 1

        new_width = w + pad

        tx = pad if shear < 0 else 0

        matrix = np.float32([
            [1, shear, tx],
            [0, 1, 0]
        ])

        sheared = cv2.warpAffine(
            binary_crop,
            matrix,
            (new_width, h),
            flags=cv2.INTER_NEAREST,
            borderValue=0
        )

        column_sums = (
            sheared
            .sum(axis=0)
            .astype(np.float64)
        )

        variance = column_sums.var()

        if variance > best_variance:

            best_variance = variance
            best_angle = angle

        angle += angle_step

    return best_angle


# ============================================================
# EXTRACT ONE FORM
# ============================================================

def extract_form(xml_path):

    root = ET.parse(
        xml_path
    ).getroot()

    # --------------------------------------------------------
    # Form metadata
    # --------------------------------------------------------

    form_id = root.attrib.get("id")

    writer_id = root.attrib.get(
        "writer-id"
    )

    image_width = parse_float(
        root.attrib.get("width")
    )

    image_height = parse_float(
        root.attrib.get("height")
    )

    skew = parse_float(
        root.attrib.get("skew")
    )

    lines = root.findall(
        "./handwritten-part/line"
    )

    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path = find_image(
        form_id
    )

    if image_path is None:

        print(
            f"Image not found for {form_id}"
        )

        return None

    # --------------------------------------------------------
    # Load and binarize image
    # --------------------------------------------------------

    binary = None

    try:

        import cv2

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15
        )

    except ImportError:

        print(
            "OpenCV is not installed. "
            "Image-based features will be NA."
        )

    except Exception as exc:

        print(
            f"Could not process "
            f"{image_path.name}: {exc}"
        )

    # --------------------------------------------------------
    # Feature containers
    # --------------------------------------------------------

    all_word_boxes = []

    component_heights = []

    baseline_rms_values = []

    spacing_cvs = []

    slants = []

    segmentation_errors = 0

    word_count = 0

    component_count = 0

    # --------------------------------------------------------
    # Overall handwriting bounding box
    #
    # IMPORTANT:
    # This comes from the IAM XML word/component coordinates.
    # It is used for Writing Area and image-region features.
    # --------------------------------------------------------

    min_x = float("inf")
    max_x = -float("inf")

    min_y = float("inf")
    max_y = -float("inf")

    # ========================================================
    # PROCESS EACH LINE
    # ========================================================

    for line in lines:

        line_words = []

        # ----------------------------------------------------
        # Segmentation errors
        # ----------------------------------------------------

        if line.attrib.get(
            "segmentation"
        ) == "err":

            segmentation_errors += 1

        # ----------------------------------------------------
        # PROCESS WORDS
        # ----------------------------------------------------

        for word in line.findall("./word"):

            components = word.findall(
                "./cmp"
            )

            if not components:
                continue

            boxes = []

            # ------------------------------------------------
            # Read IAM component boxes
            # ------------------------------------------------

            for cmp_node in components:

                x = parse_float(
                    cmp_node.attrib.get("x")
                )

                y = parse_float(
                    cmp_node.attrib.get("y")
                )

                width = parse_float(
                    cmp_node.attrib.get("width")
                )

                height = parse_float(
                    cmp_node.attrib.get("height")
                )

                if None in (
                    x,
                    y,
                    width,
                    height
                ):
                    continue

                boxes.append(
                    (
                        x,
                        y,
                        width,
                        height
                    )
                )

                component_heights.append(
                    height
                )

                component_count += 1

            if not boxes:
                continue

            # ------------------------------------------------
            # Word bounding box
            # ------------------------------------------------

            x1 = min(
                box[0]
                for box in boxes
            )

            y1 = min(
                box[1]
                for box in boxes
            )

            x2 = max(
                box[0] + box[2]
                for box in boxes
            )

            y2 = max(
                box[1] + box[3]
                for box in boxes
            )

            # ------------------------------------------------
            # Baseline point
            #
            # Use MEDIAN of component bottoms rather than
            # the maximum bottom.
            #
            # This reduces the effect of descenders such as
            # g, y, p, j and q.
            # ------------------------------------------------

            component_bottoms = [
                box[1] + box[3]
                for box in boxes
            ]

            word_bottom = median(
                component_bottoms
            )

            word_box = {

                "x": x1,

                "y": y1,

                "width": x2 - x1,

                "height": y2 - y1,

                "bottom": word_bottom,

                "center_x":
                    (x1 + x2) / 2.0
            }

            line_words.append(
                word_box
            )

            all_word_boxes.append(
                word_box
            )

            word_count += 1

            # ------------------------------------------------
            # Update overall handwriting bounding box
            # ------------------------------------------------

            min_x = min(
                min_x,
                x1
            )

            max_x = max(
                max_x,
                x2
            )

            min_y = min(
                min_y,
                y1
            )

            max_y = max(
                max_y,
                y2
            )

        # ====================================================
        # WORD SPACING CV
        # ====================================================

        line_words.sort(
            key=lambda word:
                word["x"]
        )

        gaps = []

        for first, second in zip(
            line_words,
            line_words[1:]
        ):

            gap = (
                second["x"]
                - (
                    first["x"]
                    + first["width"]
                )
            )

            if gap > 0:
                gaps.append(gap)

        line_spacing_cv = cv(
            gaps
        )

        if line_spacing_cv is not None:

            spacing_cvs.append(
                line_spacing_cv
            )

        # ====================================================
        # BASELINE DEVIATION
        # ====================================================

        if len(line_words) >= 2:

            xs = [
                word["center_x"]
                for word in line_words
            ]

            ys = [
                word["bottom"]
                for word in line_words
            ]

            x_bar = mean(xs)

            y_bar = mean(ys)

            denominator = sum(
                (
                    x - x_bar
                ) ** 2
                for x in xs
            )

            if denominator > 0:

                slope = sum(
                    (
                        x - x_bar
                    )
                    *
                    (
                        y - y_bar
                    )

                    for x, y
                    in zip(xs, ys)

                ) / denominator

                intercept = (
                    y_bar
                    - slope * x_bar
                )

                residuals = [

                    y
                    - (
                        slope * x
                        + intercept
                    )

                    for x, y
                    in zip(xs, ys)
                ]

                rms = math.sqrt(
                    mean(
                        [
                            r * r
                            for r in residuals
                        ]
                    )
                )

                baseline_rms_values.append(
                    rms
                )

        # ====================================================
        # SLANT ANGLE
        # ====================================================

        if (
            binary is not None
            and line_words
        ):

            line_x_min = int(
                max(
                    0,
                    min(
                        word["x"]
                        for word
                        in line_words
                    )
                )
            )

            line_x_max = int(
                min(
                    binary.shape[1],
                    max(
                        word["x"]
                        + word["width"]
                        for word
                        in line_words
                    )
                )
            )

            line_y_min = int(
                max(
                    0,
                    min(
                        word["y"]
                        for word
                        in line_words
                    )
                )
            )

            line_y_max = int(
                min(
                    binary.shape[0],
                    max(
                        word["y"]
                        + word["height"]
                        for word
                        in line_words
                    )
                )
            )

            if (
                line_x_max > line_x_min
                and line_y_max > line_y_min
            ):

                line_crop = binary[
                    line_y_min:line_y_max,
                    line_x_min:line_x_max
                ]

                try:

                    angle = estimate_line_slant(
                        line_crop
                    )

                    if angle is not None:

                        slants.append(
                            angle
                        )

                except Exception as exc:

                    print(
                        f"Slant estimation failed "
                        f"for a line in {form_id}: "
                        f"{exc}"
                    )

    # ========================================================
    # AGGREGATE XML-BASED FEATURES
    # ========================================================

    baseline_deviation = (

        mean(
            baseline_rms_values
        )

        if baseline_rms_values

        else "NA"
    )

    word_spacing_cv = (

        mean(
            spacing_cvs
        )

        if spacing_cvs

        else "NA"
    )

    # Character Height CV
    character_height_cv = (

        cv(
            component_heights
        )

        if component_heights

        else "NA"
    )

    average_word_height = (

        mean(
            [
                word["height"]
                for word
                in all_word_boxes
            ]
        )

        if all_word_boxes

        else "NA"
    )

    average_word_width = (

        mean(
            [
                word["width"]
                for word
                in all_word_boxes
            ]
        )

        if all_word_boxes

        else "NA"
    )

    slant_angle = (

        mean(slants)

        if slants

        else "NA"
    )

    # ========================================================
    # IMAGE-BASED FEATURES
    # ========================================================

    stroke_density = "NA"

    writing_area = "NA"

    connected_component_count = "NA"

    if (
        binary is not None
        and min_x != float("inf")
    ):

        try:

            import cv2
            import numpy as np

            # ------------------------------------------------
            # Handwriting bounding box
            #
            # IMPORTANT:
            # This is based on the IAM XML coordinates,
            # not on every non-zero pixel in the page.
            # ------------------------------------------------

            x1 = max(
                0,
                int(
                    math.floor(min_x)
                )
            )

            x2 = min(
                binary.shape[1],
                int(
                    math.ceil(max_x)
                )
            )

            y1 = max(
                0,
                int(
                    math.floor(min_y)
                )
            )

            y2 = min(
                binary.shape[0],
                int(
                    math.ceil(max_y)
                )
            )

            if (
                x2 > x1
                and y2 > y1
            ):

                handwriting_crop = binary[
                    y1:y2,
                    x1:x2
                ]

                region_height, region_width = (
                    handwriting_crop.shape
                )

                region_area = (
                    region_width
                    * region_height
                )

                total_page_pixels = (
                    binary.shape[0]
                    * binary.shape[1]
                )

                if region_area > 0:

                    # ====================================================
                    # 6. STROKE DENSITY
                    # ====================================================
                    #
                    # Ink pixels inside handwriting region
                    # divided by handwriting-region area.
                    #
                    # This measures how densely the writing region
                    # contains ink.
                    # ====================================================

                    crop_ink_pixels = int(
                        np.sum(
                            handwriting_crop == 255
                        )
                    )

                    stroke_density = (
                        crop_ink_pixels
                        / region_area
                    )

                    # ====================================================
                    # 8. WRITING AREA
                    # ====================================================
                    #
                    # Handwriting bounding-box area
                    # divided by whole-page area.
                    #
                    # This measures how much of the page is occupied
                    # by the handwriting block.
                    # ====================================================

                    writing_area = (
                        region_area
                        / total_page_pixels
                    )

                    # ====================================================
                    # 9. CONNECTED COMPONENT COUNT
                    # ====================================================
                    #
                    # Use actual image-pixel connectivity.
                    #
                    # IMPORTANT:
                    # This is NOT the IAM <cmp> count.
                    #
                    # Steps:
                    #   1. handwriting crop
                    #   2. median blur
                    #   3. morphological opening
                    #   4. 8-connected component labeling
                    #   5. remove components smaller than 10 pixels
                    # ====================================================

                    blurred = cv2.medianBlur(
                        handwriting_crop,
                        3
                    )

                    kernel = np.ones(
                        (2, 2),
                        np.uint8
                    )

                    cleaned = cv2.morphologyEx(
                        blurred,
                        cv2.MORPH_OPEN,
                        kernel
                    )

                    (
                        num_labels,
                        labels,
                        stats,
                        _
                    ) = cv2.connectedComponentsWithStats(
                        cleaned,
                        connectivity=8
                    )

                    # Label 0 = background
                    areas = stats[
                        1:,
                        cv2.CC_STAT_AREA
                    ]

                    # Ignore tiny noise
                    valid_areas = areas[
                        areas >= 10
                    ]

                    connected_component_count = int(
                        len(valid_areas)
                    )

        except Exception as exc:

            print(
                f"Could not compute image-based "
                f"features for {form_id}: {exc}"
            )

    # ========================================================
    # RETURN FINAL METADATA ROW
    # ========================================================

    return {

        "sample_id":
            form_id,

        "form_id":
            form_id,

        "writer_id":
            writer_id,

        "image_path":
            str(
                image_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "image_width":
            (
                int(image_width)
                if image_width is not None
                else "NA"
            ),

        "image_height":
            (
                int(image_height)
                if image_height is not None
                else "NA"
            ),

        "skew":
            (
                skew
                if skew is not None
                else "NA"
            ),

        # ----------------------------------------------------
        # ORIGINAL 9 FEATURES
        # ----------------------------------------------------

        "baseline_deviation":
            baseline_deviation,

        "word_spacing_cv":
            word_spacing_cv,

        "character_height_cv":
            character_height_cv,

        "average_word_height":
            average_word_height,

        "average_word_width":
            average_word_width,

        "stroke_density":
            stroke_density,

        "slant_angle":
            slant_angle,

        "writing_area":
            writing_area,

        "connected_component_count":
            connected_component_count,

        # ----------------------------------------------------
        # SANITY-CHECK METADATA
        # ----------------------------------------------------

        "num_lines":
            len(lines),

        "num_words":
            word_count,

        "num_components":
            component_count,

        "segmentation_error_count":
            segmentation_errors,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check IAM XML directory
    # --------------------------------------------------------

    if not XML_DIR.exists():

        raise FileNotFoundError(
            f"XML folder not found:\n{XML_DIR}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image folder not found:\n{IMAGE_DIR}"
        )

    # --------------------------------------------------------
    # Select 10 forms from different writers
    # --------------------------------------------------------

    # Select every XML file that has a corresponding image.
    xml_files = []
    for xml_path in sorted(XML_DIR.glob("*.xml")):
    
        form_id = xml_path.stem

        image_path = find_image(form_id)
        

        if image_path is not None:

            xml_files.append(xml_path)
        else:

            print(
                f"Skipping {form_id}: "
                f"corresponding image not found."
                
            )
    print(
        f"\nFound {len(xml_files)} usable IAM forms."
    )


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    rows = []

    for index, xml_path in enumerate(xml_files, start=1):

        print(
            f"[{index}/{len(xml_files)}] "
            f"Processing {xml_path.name} ..."
        )

        try:

            row = extract_form(
                xml_path
            )

            if row is not None:

                rows.append(row)

        except Exception as exc:

            print(
                f"Error processing "
                f"{xml_path.name}: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # CSV column order
    # --------------------------------------------------------

    fieldnames = [

        "sample_id",
        "form_id",
        "writer_id",

        "image_path",
        "image_width",
        "image_height",
        "skew",

        "baseline_deviation",
        "word_spacing_cv",
        "character_height_cv",
        "average_word_height",
        "average_word_width",
        "stroke_density",
        "slant_angle",
        "writing_area",
        "connected_component_count",

        "num_lines",
        "num_words",
        "num_components",
        "segmentation_error_count",
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

        writer.writerows(
            rows
        )

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n===================================")
    print("Full IAM metadata extraction done.")
    print("===================================")
    print(f"Created: {OUTPUT_CSV}")
    print(f"Samples processed: {len(rows)}")


if __name__ == "__main__":
    main()