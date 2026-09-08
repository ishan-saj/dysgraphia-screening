from pathlib import Path
import argparse
import base64
import html
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

MASTER_FILE = METADATA_DIR / "master_anomaly_analysis.csv"
ANOMALY_FILE = METADATA_DIR / "anomaly_scores.csv"
WORD_ANOMALY_FILE = METADATA_DIR / "all_samples_word_anomaly.csv"
WORD_SUMMARY_FILE = METADATA_DIR / "all_samples_word_summary.csv"

METADATA_ALL_FILE = METADATA_DIR / "metadata_all.csv"

OUTPUT_DIR = METADATA_DIR / "final_reports"

COMPARISON_CHART = (
    METADATA_DIR / "resnet_vs_word_anomaly.png"
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

SAMPLE_FEATURES = {

    "skew": (
        "Skew",
        "Overall geometric skew/orientation measurement of the handwriting image.",
        "degrees / image measurement"
    ),

    "baseline_deviation": (
        "Baseline Deviation",
        "Measures how much handwriting deviates from its estimated writing baseline.",
        "pixels / image coordinate"
    ),

    "word_spacing_cv": (
        "Word Spacing CV",
        "Coefficient of variation of spacing between words. Higher values indicate more variable spacing.",
        "coefficient of variation"
    ),

    "character_height_cv": (
        "Character Height CV",
        "Variation in character heights within the handwriting sample.",
        "coefficient of variation"
    ),

    "average_word_height": (
        "Average Word Height",
        "Average height of detected handwriting words.",
        "pixels"
    ),

    "average_word_width": (
        "Average Word Width",
        "Average width of detected handwriting words.",
        "pixels"
    ),

    "stroke_density": (
        "Stroke Density",
        "Density of handwriting ink/strokes within the writing region.",
        "ratio"
    ),

    "slant_angle": (
        "Slant Angle",
        "Estimated overall inclination/slant of the handwriting strokes.",
        "degrees"
    ),

    "writing_area": (
        "Writing Area",
        "Fraction of the image occupied by detected handwriting.",
        "ratio"
    ),

    "connected_component_count": (
        "Connected Component Count",
        "Number of connected handwriting components detected in the sample.",
        "count"
    ),

    "num_lines": (
        "Number of Lines",
        "Number of detected handwriting lines.",
        "count"
    ),

    "num_words": (
        "Number of Words",
        "Number of detected word regions.",
        "count"
    ),

    "num_components": (
        "Number of Components",
        "Number of connected components associated with the handwriting.",
        "count"
    ),

    "segmentation_error_count": (
        "Segmentation Errors",
        "Number of regions marked as having segmentation errors.",
        "count"
    ),
}


WORD_FEATURES = {

    "aspect_ratio": (
        "Aspect Ratio",
        "Ratio between the width and height of the word region."
    ),

    "ink_aspect_ratio": (
        "Ink Aspect Ratio",
        "Ratio between the width and height of the actual handwriting ink region."
    ),

    "ink_density": (
        "Ink Density",
        "Proportion of the word region occupied by handwriting ink."
    ),

    "centroid_x_ratio": (
        "Horizontal Ink Centroid",
        "Relative horizontal center of the handwriting ink."
    ),

    "centroid_y_ratio": (
        "Vertical Ink Centroid",
        "Relative vertical center of the handwriting ink."
    ),

    "largest_component_ratio": (
        "Largest Component Ratio",
        "Fraction of handwriting ink belonging to the largest connected component."
    ),

    "horizontal_projection_std": (
        "Horizontal Projection Standard Deviation",
        "Measures variation of handwriting ink distribution across horizontal rows."
    ),

    "vertical_projection_std": (
        "Vertical Projection Standard Deviation",
        "Measures variation of handwriting ink distribution across vertical columns."
    ),

    "lower_ink_ratio": (
        "Lower Ink Ratio",
        "Proportion of handwriting ink located in the lower part of the word region."
    ),
}


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return html.escape(str(value))


def format_number(value, digits=3):

    try:

        if pd.isna(value):
            return "-"

        return f"{float(value):.{digits}f}"

    except (ValueError, TypeError):

        return safe_text(value)


def image_to_base64(path):

    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        return None

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(
        path.suffix.lower(),
        "image/png"
    )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


def find_image(path_value):

    if path_value is None:
        return None

    try:

        if pd.isna(path_value):
            return None

    except Exception:
        pass

    path = Path(str(path_value))

    candidates = [
        path,
        PROJECT_ROOT / path,
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    # --------------------------------------------------------
    # Filename fallback
    # --------------------------------------------------------

    filename = path.name

    raw_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    if raw_dir.exists():

        matches = list(
            raw_dir.rglob(filename)
        )

        if matches:
            return matches[0]

    # --------------------------------------------------------
    # Metadata fallback
    # --------------------------------------------------------

    return None


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    required_files = [

        MASTER_FILE,
        ANOMALY_FILE,
        WORD_ANOMALY_FILE,
        WORD_SUMMARY_FILE,
        METADATA_ALL_FILE,

    ]

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )

    master_df = pd.read_csv(
        MASTER_FILE
    )

    anomaly_df = pd.read_csv(
        ANOMALY_FILE
    )

    word_df = pd.read_csv(
        WORD_ANOMALY_FILE
    )

    summary_df = pd.read_csv(
        WORD_SUMMARY_FILE
    )

    metadata_df = pd.read_csv(
        METADATA_ALL_FILE
    )

    return (
        master_df,
        anomaly_df,
        word_df,
        summary_df,
        metadata_df,
    )


# ============================================================
# SAMPLE INFORMATION
# ============================================================

def get_sample_row(
    sample_id,
    master_df,
    anomaly_df,
    summary_df,
    metadata_df
):

    result = {}

    # --------------------------------------------------------
    # Original metadata = SOURCE OF TRUTH
    # --------------------------------------------------------

    metadata = metadata_df[
        metadata_df["sample_id"].astype(str)
        == sample_id
    ]

    if len(metadata):

        for column in metadata_df.columns:

            result[column] = (
                metadata.iloc[0][column]
            )

    # --------------------------------------------------------
    # Master anomaly analysis
    # --------------------------------------------------------

    master = master_df[
        master_df["sample_id"].astype(str)
        == sample_id
    ]

    if len(master):

        for column in master_df.columns:

            result[column] = (
                master.iloc[0][column]
            )

    # --------------------------------------------------------
    # ResNet anomaly
    # --------------------------------------------------------

    anomaly = anomaly_df[
        anomaly_df["sample_id"].astype(str)
        == sample_id
    ]

    if len(anomaly):

        for column in anomaly_df.columns:

            result[column] = (
                anomaly.iloc[0][column]
            )

    # --------------------------------------------------------
    # Word summary
    # --------------------------------------------------------

    summary = summary_df[
        summary_df["sample_id"].astype(str)
        == sample_id
    ]

    if len(summary):

        for column in summary_df.columns:

            result[column] = (
                summary.iloc[0][column]
            )

    return result


# ============================================================
# SAMPLE-LEVEL OPENCV TABLE
# ============================================================

def create_sample_feature_table(row):

    rows = ""

    for column, (
        name,
        description,
        unit
    ) in SAMPLE_FEATURES.items():

        if column not in row:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        rows += f"""
        <tr>

            <td>
                <strong>{safe_text(name)}</strong>
            </td>

            <td>
                {format_number(value, 3)}
            </td>

            <td>
                {safe_text(unit)}
            </td>

            <td>
                {safe_text(description)}
            </td>

        </tr>
        """

    return f"""
    <table>

        <thead>

            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>Unit</th>
                <th>What it measures</th>
            </tr>

        </thead>

        <tbody>

            {rows}

        </tbody>

    </table>
    """


# ============================================================
# WORD EXPLAINABILITY DATA
# ============================================================

def load_word_explainability(
    sample_id
):

    folder = (
        METADATA_DIR
        / f"{sample_id}_word_explainability"
    )

    report_file = (
        folder
        / "word_explainability_report.csv"
    )

    if not report_file.exists():

        print(
            "WARNING: Word explainability CSV not found:"
        )

        print(report_file)

        return (
            None,
            folder if folder.exists() else None
        )

    explanation_df = pd.read_csv(
        report_file
    )

    return (
        explanation_df,
        folder
    )


# ============================================================
# WORD DATA
# ============================================================

def get_top_words(
    sample_id,
    word_df,
    explanation_df=None,
    number=10
):

    # --------------------------------------------------------
    # Filter sample
    # --------------------------------------------------------

    if "form_id" in word_df.columns:

        sample_words = word_df[
            word_df["form_id"].astype(str)
            == sample_id
        ].copy()

    elif "sample_id" in word_df.columns:

        sample_words = word_df[
            word_df["sample_id"].astype(str)
            == sample_id
        ].copy()

    else:

        return pd.DataFrame()

    if len(sample_words) == 0:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Sort anomaly score
    # --------------------------------------------------------

    sample_words = sample_words.sort_values(
        "word_anomaly_score",
        ascending=False
    ).head(number).copy()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Merge with word_explainability_report.csv
    #
    # This attaches:
    # original_crop
    # gradcam_image
    # comparison_image
    # rank
    # high_activation_percentage
    #
    # to each anomalous word.
    # --------------------------------------------------------

    if (
        explanation_df is not None
        and len(explanation_df) > 0
        and "word_id" in sample_words.columns
        and "word_id" in explanation_df.columns
    ):

        explanation_columns = [
            "word_id",
            "rank",
            "word",
            "high_activation_percentage",
            "original_crop",
            "gradcam_image",
            "comparison_image",
        ]

        available_columns = [
            column
            for column in explanation_columns
            if column in explanation_df.columns
        ]

        explanation_merge = (
            explanation_df[
                available_columns
            ]
            .drop_duplicates(
                subset=["word_id"]
            )
        )

        sample_words = sample_words.merge(
            explanation_merge,
            on="word_id",
            how="left",
            suffixes=("", "_explanation")
        )

        # ----------------------------------------------------
        # Use explanation rank if available
        # ----------------------------------------------------

        if "rank" in sample_words.columns:

            sample_words = sample_words.sort_values(
                "rank",
                na_position="last"
            )

    return sample_words.reset_index(
        drop=True
    )


# ============================================================
# WORD FEATURE TABLE
# ============================================================

def create_word_feature_table(row):

    rows = ""

    for column, (
        name,
        description
    ) in WORD_FEATURES.items():

        if column not in row:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        z_column = f"{column}_z"

        z_value = row.get(
            z_column,
            None
        )

        z_text = "-"

        if (
            z_value is not None
            and not pd.isna(z_value)
        ):

            z_text = format_number(
                z_value,
                2
            )

        rows += f"""
        <tr>

            <td>
                <strong>
                    {safe_text(name)}
                </strong>
            </td>

            <td>
                {format_number(value, 4)}
            </td>

            <td>
                {z_text}
            </td>

            <td>
                {safe_text(description)}
            </td>

        </tr>
        """

    return f"""
    <table class="feature-table">

        <thead>

            <tr>

                <th>OpenCV Feature</th>
                <th>Value</th>
                <th>Robust Z-score</th>
                <th>Meaning</th>

            </tr>

        </thead>

        <tbody>

            {rows}

        </tbody>

    </table>
    """


# ============================================================
# WORD IMAGE PATHS
# ============================================================

def get_word_image_paths(
    row,
    folder,
    rank
):

    paths = {

        "original": None,

        "gradcam": None,

        "comparison": None,

    }

    # --------------------------------------------------------
    # PRIMARY METHOD:
    # Read exact paths from word_explainability_report.csv
    # --------------------------------------------------------

    for key, column in [

        ("original", "original_crop"),

        ("gradcam", "gradcam_image"),

        ("comparison", "comparison_image"),

    ]:

        if column not in row:
            continue

        value = row[column]

        if value is None:
            continue

        try:

            if pd.isna(value):
                continue

        except Exception:

            pass

        found = find_image(
            value
        )

        if found:

            paths[key] = found

    # --------------------------------------------------------
    # FALLBACK:
    # Search inside explainability folder
    # --------------------------------------------------------

    if folder and folder.exists():

        word = str(
            row.get(
                "transcription",
                row.get(
                    "word",
                    "word"
                )
            )
        )

        # Match the naming used by word_gradcam.py
        safe_word = (
            word
            .replace("/", "_")
            .replace("\\", "_")
        )

        search_patterns = {

            "original": [
                f"{rank:02d}_{safe_word}_*_original.png",
                f"{rank:02d}_{safe_word}_original.png",
            ],

            "gradcam": [
                f"{rank:02d}_{safe_word}_*_gradcam.png",
                f"{rank:02d}_{safe_word}_gradcam.png",
            ],

            "comparison": [
                f"{rank:02d}_{safe_word}_*_comparison.png",
                f"{rank:02d}_{safe_word}_comparison.png",
            ],

        }

        for key, patterns in search_patterns.items():

            if paths[key] is not None:
                continue

            for pattern in patterns:

                matches = list(
                    folder.rglob(pattern)
                )

                if matches:

                    paths[key] = matches[0]

                    break

    return paths


# ============================================================
# WORD EXPLANATION CARD
# ============================================================

def create_word_section(
    rank,
    row,
    explainability_folder
):

    word = row.get(
        "transcription",
        row.get(
            "word",
            "Unknown"
        )
    )

    score = row.get(
        "word_anomaly_score",
        0
    )

    main_feature = row.get(
        "main_anomaly_feature",
        row.get(
            "main_feature",
            "Unknown"
        )
    )

    main_z = row.get(
        "main_feature_z",
        0
    )

    feature_name = WORD_FEATURES.get(
        main_feature,
        (
            str(main_feature)
            .replace("_", " ")
            .title(),

            "Handcrafted handwriting feature."
        )
    )[0]

    feature_description = WORD_FEATURES.get(
        main_feature,
        (
            "",

            "This feature measures a measurable "
            "property of the handwriting region."
        )
    )[1]

    # --------------------------------------------------------
    # IMAGE PATHS
    # --------------------------------------------------------

    paths = get_word_image_paths(
        row,
        explainability_folder,
        rank
    )

    # --------------------------------------------------------
    # EXACT THREE-COLUMN VISUALIZATION
    # --------------------------------------------------------

    image_boxes = ""

    for title, key in [

        ("Original Word", "original"),

        ("Grad-CAM", "gradcam"),

        ("Comparison", "comparison"),

    ]:

        path = paths[key]

        if path:

            b64 = image_to_base64(
                path
            )

            if b64:

                image_boxes += f"""

                <div class="visual-box">

                    <h3>
                        {title}
                    </h3>

                    <img
                        src="{b64}"
                        class="word-visual"
                        alt="{title}"
                    >

                </div>

                """

            else:

                image_boxes += f"""

                <div class="visual-box">

                    <h3>
                        {title}
                    </h3>

                    <div class="missing-image">
                        Image could not be loaded.
                    </div>

                </div>

                """

        else:

            image_boxes += f"""

            <div class="visual-box">

                <h3>
                    {title}
                </h3>

                <div class="missing-image">
                    Image not found.
                </div>

            </div>

            """

    # --------------------------------------------------------
    # Grad-CAM activation
    # --------------------------------------------------------

    activation = row.get(
        "high_activation_percentage",
        None
    )

    activation_html = ""

    if (
        activation is not None
        and not pd.isna(activation)
    ):

        activation_html = f"""

        <p class="activation-note">

            <strong>
                Grad-CAM highlighted area:
            </strong>

            {format_number(activation, 2)}%

            <br>

            <span>
                This percentage is a visualization
                threshold measurement. It is not an
                abnormality severity score.
            </span>

        </p>

        """

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation = row.get(
        "explanation",
        None
    )

    if explanation is not None:

        explanation_text = safe_text(
            explanation
        )

    else:

        explanation_text = f"""
        The word has an anomaly score of
        <strong>{format_number(score, 2)}</strong>.
        The primary measured feature is
        <strong>{safe_text(feature_name)}</strong>,
        with a robust z-score of
        <strong>{format_number(main_z, 2)}</strong>.
        """

    # --------------------------------------------------------
    # COMPLETE WORD CARD
    # --------------------------------------------------------

    return f"""

    <section class="word-card">

        <div class="word-header">

            <div>

                <span class="word-rank">
                    #{rank}
                </span>

                <span class="word-name">
                    {safe_text(word)}
                </span>

            </div>

            <div class="word-score">

                Anomaly Score:
                <strong>
                    {format_number(score, 3)}
                </strong>

            </div>

        </div>


        <div class="word-main-info">

            <div>

                <strong>
                    Main contributing feature:
                </strong>

                {safe_text(feature_name)}

            </div>

            <div>

                <strong>
                    Robust Z-score:
                </strong>

                {format_number(main_z, 2)}

            </div>

        </div>


        <h3>
            Word-Level OpenCV Feature Analysis
        </h3>

        {create_word_feature_table(row)}


        <h2 class="visual-title">
            Visual Explainability
        </h2>


        <div class="visual-grid">

            {image_boxes}

        </div>


        {activation_html}


        <div class="gradcam-legend">

            <div class="legend-bar"></div>

            <div class="legend-labels">

                <span>
                    Low contribution
                </span>

                <span>
                    Moderate
                </span>

                <span>
                    High contribution
                </span>

            </div>

            <p>
                Blue = lower model contribution |
                Red = highest model contribution
            </p>

        </div>


        <div class="explanation-box">

            <h3>
                Explanation
            </h3>

            <p>
                {explanation_text}
            </p>

            <p>

                The Grad-CAM highlights image regions
                that contribute strongly to the ResNet50
                visual representation used during the
                anomaly analysis.

            </p>

            <p>

                These highlighted regions should be
                interpreted as model evidence for
                unusualness, not as a clinical diagnosis
                of dysgraphia.

            </p>

        </div>

    </section>

    """


# ============================================================
# GENERATE HTML
# ============================================================

def generate_html(
    sample_id,
    row,
    top_words,
    original_image,
    explainability_folder
):

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    resnet_score = row.get(
        "resnet_anomaly_score",
        row.get(
            "anomaly_score_0_100",
            0
        )
    )

    mean_word = row.get(
        "mean_word_anomaly",
        0
    )

    max_word = row.get(
        "maximum_word_anomaly",
        0
    )

    top5 = row.get(
        "top_5_mean_anomaly",
        0
    )

    combined = row.get(
        "combined_anomaly_score",
        row.get(
            "combined_score",
            row.get(
                "combined_rank_score",
                0
            )
        )
    )

    agreement = row.get(
        "agreement",
        "Not available"
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original_b64 = image_to_base64(
        original_image
    )

    if original_b64:

        original_html = f"""

        <img
            src="{original_b64}"
            class="main-image"
            alt="Original handwriting"
        >

        """

    else:

        original_html = """

        <div class="missing">
            Original handwriting image could not be found.
        </div>

        """

    # --------------------------------------------------------
    # Comparison chart
    # --------------------------------------------------------

    comparison_html = ""

    if COMPARISON_CHART.exists():

        b64 = image_to_base64(
            COMPARISON_CHART
        )

        if b64:

            comparison_html = f"""

            <img
                src="{b64}"
                class="comparison-chart"
                alt="ResNet versus word anomaly comparison"
            >

            """

    # --------------------------------------------------------
    # Word sections
    # --------------------------------------------------------

    word_sections = ""

    if len(top_words) > 0:

        for index, (_, word_row) in enumerate(
            top_words.iterrows(),
            start=1
        ):

            # Use Grad-CAM rank when available
            rank_value = word_row.get(
                "rank",
                index
            )

            try:

                if pd.isna(rank_value):

                    rank_value = index

                rank_value = int(
                    rank_value
                )

            except Exception:

                rank_value = index

            word_sections += create_word_section(
                rank=rank_value,
                row=word_row,
                explainability_folder=(
                    explainability_folder
                )
            )

    else:

        word_sections = """

        <div class="missing">

            No word-level anomaly data was found
            for this sample.

        </div>

        """

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    return f"""

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
Handwriting Analysis Report - {safe_text(sample_id)}
</title>


<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

* {{
    box-sizing: border-box;
}}


body {{

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f3f5f7;

    color: #202124;

    line-height: 1.6;
}}


.container {{

    max-width: 1350px;

    margin: auto;

    padding: 35px;
}}


/* ==========================================================
   HEADER
   ========================================================== */

.header {{

    background: white;

    padding: 35px;

    border-radius: 15px;

    margin-bottom: 25px;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.05);
}}


h1 {{

    margin-top: 0;

    font-size: 32px;
}}


h2 {{

    margin-top: 45px;

    border-bottom: 2px solid #ddd;

    padding-bottom: 10px;
}}


h3 {{

    margin-top: 20px;
}}


h4 {{
    margin-bottom: 8px;
}}


/* ==========================================================
   DISCLAIMER
   ========================================================== */

.disclaimer {{

    background: #fff3cd;

    border-left: 5px solid #e0a800;

    padding: 18px;

    border-radius: 8px;

    margin-top: 20px;
}}


/* ==========================================================
   SUMMARY
   ========================================================== */

.summary-grid {{

    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(190px, 1fr));

    gap: 15px;
}}


.metric {{

    background: white;

    padding: 22px;

    border-radius: 12px;

    text-align: center;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
}}


.metric-label {{

    color: #666;

    font-size: 14px;
}}


.metric-value {{

    font-size: 27px;

    font-weight: bold;

    margin-top: 5px;
}}


/* ==========================================================
   IMAGES
   ========================================================== */

.main-image {{

    max-width: 100%;

    max-height: 850px;

    display: block;

    margin: 20px auto;

    border: 1px solid #ddd;

    border-radius: 10px;
}}


.comparison-chart {{

    width: 100%;

    max-width: 1100px;

    display: block;

    margin: 25px auto;

    border-radius: 10px;
}}


/* ==========================================================
   METHOD BOX
   ========================================================== */

.method-box {{

    background: white;

    padding: 25px;

    border-radius: 12px;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.04);
}}


/* ==========================================================
   ARCHITECTURE
   ========================================================== */

.architecture {{

    font-family: monospace;

    background: #f7f7f7;

    padding: 20px;

    border-radius: 10px;

    white-space: pre-wrap;

    overflow-x: auto;
}}


/* ==========================================================
   TABLES
   ========================================================== */

table {{

    width: 100%;

    border-collapse: collapse;

    margin-top: 15px;

    background: white;
}}


th {{

    text-align: left;

    background: #f0f2f4;

    padding: 12px;

    border-bottom: 2px solid #ddd;
}}


td {{

    padding: 11px;

    border-bottom: 1px solid #e5e5e5;

    vertical-align: top;
}}


.feature-table {{

    font-size: 13px;
}}


/* ==========================================================
   WORD CARD
   ========================================================== */

.word-card {{

    background: white;

    padding: 28px;

    border-radius: 14px;

    margin-top: 35px;

    box-shadow:
        0 2px 8px rgba(0,0,0,0.06);
}}


.word-header {{

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 20px;

    flex-wrap: wrap;

    border-bottom: 1px solid #ddd;

    padding-bottom: 15px;
}}


.word-rank {{

    font-size: 18px;

    font-weight: bold;

    margin-right: 12px;
}}


.word-name {{

    font-size: 27px;

    font-weight: bold;
}}


.word-score {{

    font-size: 16px;
}}


.word-main-info {{

    display: flex;

    gap: 35px;

    flex-wrap: wrap;

    background: #f7f8fa;

    padding: 15px;

    border-radius: 8px;

    margin-top: 20px;
}}


/* ==========================================================
   VISUAL EXPLAINABILITY
   ========================================================== */

.visual-title {{

    margin-top: 35px;

    border: none;

    padding: 0;
}}


.visual-grid {{

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 25px;

    margin-top: 20px;

    align-items: start;
}}


.visual-box {{

    text-align: center;
}}


.visual-box h3 {{

    margin-top: 0;

    margin-bottom: 12px;

    font-size: 19px;
}}


.word-visual {{

    width: 100%;

    max-height: 360px;

    object-fit: contain;

    background: white;

    border: 1px solid #d8d8d8;

    border-radius: 8px;

    padding: 3px;
}}


/* ==========================================================
   GRAD-CAM LEGEND
   ========================================================== */

.gradcam-legend {{

    width: 100%;

    max-width: 700px;

    margin: 25px auto 10px;

    text-align: center;
}}


.legend-bar {{

    height: 18px;

    border-radius: 4px;

    background:
        linear-gradient(
            to right,
            blue,
            cyan,
            lime,
            yellow,
            red
        );
}}


.legend-labels {{

    display: flex;

    justify-content: space-between;

    font-size: 12px;

    margin-top: 5px;
}}


.gradcam-legend p {{

    font-size: 12px;

    color: #666;

    margin-top: 5px;
}}


/* ==========================================================
   ACTIVATION NOTE
   ========================================================== */

.activation-note {{

    background: #f7f8fa;

    padding: 12px;

    border-radius: 7px;

    margin-top: 20px;

    font-size: 13px;
}}


.activation-note span {{

    color: #666;
}}


/* ==========================================================
   EXPLANATION
   ========================================================== */

.explanation-box {{

    background: #f7f8fa;

    padding: 20px;

    border-radius: 9px;

    margin-top: 25px;
}}


/* ==========================================================
   MISSING
   ========================================================== */

.missing-image {{

    background: #eeeeee;

    color: #777;

    padding: 50px 15px;

    border-radius: 8px;

    border: 1px solid #ddd;
}}


.missing {{

    background: #eeeeee;

    padding: 30px;

    text-align: center;

    border-radius: 10px;
}}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {{

    background: white;

    padding: 25px;

    margin-top: 50px;

    border-radius: 12px;

    color: #666;
}}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 900px) {{

    .visual-grid {{

        grid-template-columns:
            1fr;

    }}

}}


@media print {{

    body {{
        background: white;
    }}

    .container {{
        max-width: none;
    }}

    .word-card {{
        page-break-inside: avoid;
    }}

}}

</style>

</head>


<body>


<div class="container">


<!-- ===================================================== -->
<!-- HEADER -->
<!-- ===================================================== -->

<div class="header">

<h1>
Early Dysgraphia Detection
</h1>

<h2>
Comprehensive Handwriting Explainability Report
</h2>

<h3>
Sample: {safe_text(sample_id)}
</h3>

<div class="disclaimer">

<strong>Research disclaimer:</strong>

This system performs handwriting anomaly analysis.
The scores indicate how unusual measured handwriting
characteristics are relative to the reference data.

They do <strong>not</strong> represent a clinical
probability or diagnosis of dysgraphia.

Clinical assessment requires appropriately labelled
clinical data and evaluation by qualified professionals.

</div>

</div>


<!-- ===================================================== -->
<!-- 1. EXECUTIVE SUMMARY -->
<!-- ===================================================== -->

<h2>
1. Executive Summary
</h2>


<div class="summary-grid">


<div class="metric">

<div class="metric-label">
ResNet Anomaly Score
</div>

<div class="metric-value">
{format_number(resnet_score, 2)}
</div>

<div class="metric-label">
0–100 normalized
</div>

</div>


<div class="metric">

<div class="metric-label">
Mean Word Anomaly
</div>

<div class="metric-value">
{format_number(mean_word, 2)}
</div>

</div>


<div class="metric">

<div class="metric-label">
Maximum Word Anomaly
</div>

<div class="metric-value">
{format_number(max_word, 2)}
</div>

</div>


<div class="metric">

<div class="metric-label">
Top-5 Word Mean
</div>

<div class="metric-value">
{format_number(top5, 2)}
</div>

</div>


<div class="metric">

<div class="metric-label">
Combined Analysis
</div>

<div class="metric-value">
{format_number(combined, 2)}
</div>

</div>


<div class="metric">

<div class="metric-label">
Agreement
</div>

<div class="metric-value">
{safe_text(agreement)}
</div>

</div>


</div>


<!-- ===================================================== -->
<!-- 2. ORIGINAL SAMPLE -->
<!-- ===================================================== -->

<h2>
2. Original Handwriting Sample
</h2>

{original_html}


<!-- ===================================================== -->
<!-- 3. SAMPLE OPENCV -->
<!-- ===================================================== -->

<h2>
3. Sample-Level OpenCV Handwriting Measurements
</h2>

<div class="method-box">

<p>

These measurements describe the overall physical
characteristics of the handwriting sample. They are
handcrafted image-analysis features calculated from the
handwriting image and its detected structures.

</p>

{create_sample_feature_table(row)}

</div>


<!-- ===================================================== -->
<!-- 4. MODEL ARCHITECTURE -->
<!-- ===================================================== -->

<h2>
4. Feature and Model Architecture
</h2>

<div class="method-box">


<div class="architecture">

Handwriting Image
        |
        +-----------------------------+
        |                             |
        v                             v
     ResNet50                       OpenCV
     2048 features              handcrafted features
        |                             |
        |                         9 selected
        |                    word-level features
        |                             |
        |                       OpenCV Encoder
        |                             |
        |                            64
        |                             |
        +-------------+---------------+
                      |
                      v
                   Fusion
                 2112 features
                      |
                      v
                  Attention
                      |
                      v
                256-D embedding
                      |
                      v
                Anomaly Analysis

</div>


<h3>
ResNet50 Deep Features
</h3>

<p>

ResNet50 provides a
<strong>
2048-dimensional learned visual representation
</strong>
of the handwriting image.

These features capture complex visual patterns that
cannot be directly translated into one simple handwriting
measurement.

</p>


<h3>
OpenCV Handcrafted Features
</h3>

<p>

OpenCV-based image processing provides explicit,
measurable handwriting characteristics such as baseline
deviation, word spacing, character height, slant,
stroke density and writing area.

</p>


</div>


<!-- ===================================================== -->
<!-- 5. WORD-LEVEL ANALYSIS -->
<!-- ===================================================== -->

<h2>
5. Word-Level Anomaly Analysis
</h2>


<div class="method-box">

<p>

IAM word bounding boxes are used to isolate individual
words from the handwriting sample.

Each word is evaluated using nine selected handcrafted
features:

</p>


<ul>

<li>
Aspect Ratio
</li>

<li>
Ink Aspect Ratio
</li>

<li>
Ink Density
</li>

<li>
Horizontal Ink Centroid
</li>

<li>
Vertical Ink Centroid
</li>

<li>
Largest Component Ratio
</li>

<li>
Horizontal Projection Standard Deviation
</li>

<li>
Vertical Projection Standard Deviation
</li>

<li>
Lower Ink Ratio
</li>

</ul>


<p>

The word anomaly score summarizes how unusual these
measurements are relative to the reference word
distribution.

</p>


</div>


<!-- ===================================================== -->
<!-- 6. TOP 10 WORDS -->
<!-- ===================================================== -->

<h2>
6. Top 10 Anomalous Words
</h2>


{word_sections}


<!-- ===================================================== -->
<!-- 7. RESNET + GRAD-CAM -->
<!-- ===================================================== -->

<h2>
7. ResNet50 and Grad-CAM Explanation
</h2>


<div class="method-box">


<h3>
What ResNet50 does
</h3>


<p>

ResNet50 analyzes the handwriting image using learned
visual representations. Its 2048-dimensional feature
vector captures complex visual patterns that cannot be
directly translated into one simple handwriting measurement.

</p>


<h3>
What Grad-CAM does
</h3>


<p>

Grad-CAM provides a visual explanation of the deep model
by highlighting image regions that contribute strongly to
the ResNet visual representation used during anomaly
analysis.

</p>


<p>

Therefore, the Grad-CAM heatmap should be interpreted as
<strong>
deep-model visual evidence
</strong>,
while the OpenCV features provide explicit quantitative
handwriting measurements.

</p>


</div>


<!-- ===================================================== -->
<!-- 8. COMPARISON -->
<!-- ===================================================== -->

<h2>
8. Sample-Level vs Word-Level Analysis
</h2>


<div class="method-box">


<p>

The sample-level ResNet analysis evaluates the handwriting
image as a whole, while word-level analysis identifies
specific localized handwriting regions with unusual
handcrafted measurements.

</p>


{comparison_html}


</div>


<!-- ===================================================== -->
<!-- 9. INTERPRETATION -->
<!-- ===================================================== -->

<h2>
9. Overall Interpretation
</h2>


<div class="method-box">


<p>

The sample-level anomaly score indicates how unusual the
learned visual representation of this handwriting sample
is relative to the reference distribution.

</p>


<p>

The OpenCV measurements provide explicit characteristics
that can be examined independently, including baseline
deviation, spacing variation, character-height variation,
slant, stroke density and writing area.

</p>


<p>

The word-level analysis provides a more localized view by
identifying individual words whose measurable handwriting
features differ substantially from the reference
distribution.

</p>


<p>

Grad-CAM complements these measurements by showing which
regions of the handwriting contribute strongly to the deep
visual representation.

</p>


<p>

Together, these components provide a more interpretable
experimental pipeline than using a single black-box
anomaly score.

</p>


</div>


<!-- ===================================================== -->
<!-- 10. LIMITATIONS -->
<!-- ===================================================== -->

<h2>
10. Limitations
</h2>


<div class="method-box">

<ul>

<li>

The IAM dataset does not provide reliable clinical
dysgraphia labels for this experiment.

</li>

<li>

The system therefore performs anomaly detection rather
than supervised dysgraphia classification.

</li>

<li>

An unusual handwriting characteristic does not necessarily
indicate dysgraphia.

</li>

<li>

Some sample-level measurements can be influenced by
image quality, writing content and segmentation quality.

</li>

<li>

Word-level measurements can be affected by word length,
letter composition and segmentation.

</li>

<li>

Grad-CAM visualizations explain the neural representation
but are not clinical localization maps.

</li>

<li>

The combined anomaly score is a research heuristic and
has not been clinically validated.

</li>

</ul>

</div>


<!-- ===================================================== -->
<!-- 11. CONCLUSION -->
<!-- ===================================================== -->

<h2>
11. Conclusion
</h2>


<div class="method-box">


<p>

This report combines three complementary sources of
information:

</p>


<ol>

<li>

<strong>
Sample-level OpenCV measurements
</strong>

describe explicit physical handwriting characteristics.

</li>


<li>

<strong>
Word-level OpenCV anomaly analysis
</strong>

identifies localized handwriting regions with unusual
measurable characteristics.

</li>


<li>

<strong>
ResNet50 + Grad-CAM
</strong>

provides learned visual representation and visual
explainability.

</li>

</ol>


<p>

The combination provides a detailed experimental framework
for investigating unusual handwriting patterns while
maintaining a clear distinction between computational
anomaly detection and clinical diagnosis.

</p>


</div>


<!-- ===================================================== -->
<!-- FOOTER -->
<!-- ===================================================== -->

<div class="footer">

<strong>
Project:
</strong>

Early Dysgraphia Detection from Handwriting Images using
Vision Foundation Models and Fine-Tuning.

<br>
<br>

<strong>
Pipeline:
</strong>

OpenCV Handcrafted Features +
ResNet50 +
Feature Fusion +
Attention +
Anomaly Detection +
Word-Level Analysis +
Grad-CAM.

<br>
<br>

<strong>
Report type:
</strong>

Research explainability report.

</div>


</div>

</body>

</html>

"""


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Generate comprehensive handwriting report."
    )

    parser.add_argument(
        "--sample",
        required=True,
        help="Sample ID, e.g. a01-063 or p06-052"
    )

    args = parser.parse_args()

    sample_id = args.sample

    print()
    print("=" * 70)
    print(
        "COMPREHENSIVE HANDWRITING "
        "EXPLAINABILITY REPORT"
    )
    print("=" * 70)

    print(
        f"Sample: {sample_id}"
    )

    print()

    # --------------------------------------------------------
    # LOAD MAIN DATA
    # --------------------------------------------------------

    (
        master_df,
        anomaly_df,
        word_df,
        summary_df,
        metadata_df
    ) = load_data()

    # --------------------------------------------------------
    # CHECK SAMPLE
    # --------------------------------------------------------

    if not (
        master_df["sample_id"]
        .astype(str)
        .eq(sample_id)
        .any()
    ):

        raise ValueError(
            f"Sample '{sample_id}' "
            f"was not found."
        )

    # --------------------------------------------------------
    # SAMPLE INFORMATION
    # --------------------------------------------------------

    row = get_sample_row(
        sample_id,
        master_df,
        anomaly_df,
        summary_df,
        metadata_df
    )

    # --------------------------------------------------------
    # LOAD WORD GRAD-CAM EXPLANATIONS
    # --------------------------------------------------------

    (
        explanation_df,
        explainability_folder
    ) = load_word_explainability(
        sample_id
    )

    if explanation_df is not None:

        print(
            "Word Grad-CAM report found:"
        )

        print(
            f"  {explainability_folder}"
        )

        print(
            f"  Entries: {len(explanation_df)}"
        )

    else:

        print()
        print(
            "WARNING: Word Grad-CAM report "
            "was not found."
        )

        print(
            "Run word_gradcam.py first."
        )

    # --------------------------------------------------------
    # TOP WORDS
    # --------------------------------------------------------

    top_words = get_top_words(
        sample_id=sample_id,
        word_df=word_df,
        explanation_df=explanation_df,
        number=10
    )

    print()

    print(
        f"Top word results: "
        f"{len(top_words)}"
    )

    # --------------------------------------------------------
    # SHOW IMAGE CONNECTION STATUS
    # --------------------------------------------------------

    if len(top_words) > 0:

        print()
        print(
            "Word visualization files:"
        )

        for _, word in top_words.iterrows():

            word_name = word.get(
                "transcription",
                word.get(
                    "word",
                    "unknown"
                )
            )

            original = word.get(
                "original_crop",
                None
            )

            gradcam = word.get(
                "gradcam_image",
                None
            )

            comparison = word.get(
                "comparison_image",
                None
            )

            print(
                f"  {word_name}: "
                f"original={'YES' if original is not None and not pd.isna(original) else 'NO'}, "
                f"gradcam={'YES' if gradcam is not None and not pd.isna(gradcam) else 'NO'}, "
                f"comparison={'YES' if comparison is not None and not pd.isna(comparison) else 'NO'}"
            )

    # --------------------------------------------------------
    # ORIGINAL SAMPLE IMAGE
    # --------------------------------------------------------

    original_image = find_image(
        row.get("image_path")
    )

    if original_image:

        print()

        print(
            "Original image found:"
        )

        print(
            f"  {original_image}"
        )

    else:

        print()

        print(
            "WARNING: Original handwriting "
            "image not found."
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / f"{sample_id}_final_report.html"
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    document = generate_html(

        sample_id=sample_id,

        row=row,

        top_words=top_words,

        original_image=original_image,

        explainability_folder=(
            explainability_folder
        )

    )

    output_file.write_text(
        document,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "REPORT GENERATED SUCCESSFULLY"
    )
    print("=" * 70)

    print()

    print(
        f"Output:"
    )

    print(
        output_file
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()