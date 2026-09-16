from pathlib import Path
import argparse
import base64
import html
import math
import re

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"

MASTER_FILE = METADATA_DIR / "master_anomaly_analysis.csv"
ANOMALY_FILE = METADATA_DIR / "anomaly_scores.csv"
WORD_ANOMALY_FILE = METADATA_DIR / "all_samples_word_anomaly.csv"
WORD_SUMMARY_FILE = METADATA_DIR / "all_samples_word_summary.csv"
METADATA_ALL_FILE = METADATA_DIR / "metadata_all.csv"
WORDS_METADATA_FILE = METADATA_DIR / "words_metadata.csv"

OUTPUT_DIR = METADATA_DIR / "final_reports"

COMPARISON_CHART = METADATA_DIR / "resnet_vs_word_anomaly.png"


# ============================================================
# FEATURES
# ============================================================

SAMPLE_FEATURES = [
    ("skew", "Skew"),
    ("baseline_deviation", "Baseline Deviation"),
    ("word_spacing_cv", "Word Spacing CV"),
    ("character_height_cv", "Character Height CV"),
    ("average_word_height", "Average Word Height"),
    ("average_word_width", "Average Word Width"),
    ("stroke_density", "Stroke Density"),
    ("slant_angle", "Slant Angle"),
    ("writing_area", "Writing Area"),
    ("connected_component_count", "Connected Components"),
    ("num_lines", "Detected Lines"),
    ("num_words", "Detected Words"),
    ("num_components", "Segmentation Components"),
    ("segmentation_error_count", "Segmentation Errors"),
]

WORD_FEATURES = [
    ("aspect_ratio", "Aspect Ratio"),
    ("ink_aspect_ratio", "Ink Aspect Ratio"),
    ("ink_density", "Ink Density"),
    ("centroid_x_ratio", "Centroid X Ratio"),
    ("centroid_y_ratio", "Centroid Y Ratio"),
    ("largest_component_ratio", "Largest Component Ratio"),
    ("horizontal_projection_std", "Horizontal Projection Std"),
    ("vertical_projection_std", "Vertical Projection Std"),
    ("lower_ink_ratio", "Lower Ink Ratio"),
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_text(value, default="N/A"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null", ""}:
        return default

    return text


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def format_number(value, decimals=4):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def format_percent(value, decimals=1):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}%"


def safe_filename(text):
    text = safe_text(text, "unknown")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def normalize_id(value):
    return safe_text(value, "").strip()


def html_escape(value):
    return html.escape(safe_text(value, ""))


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_to_base64(path):
    if not path:
        return None

    try:
        path = Path(path)

        if not path.exists() or not path.is_file():
            return None

        suffix = path.suffix.lower()

        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix)

        if not mime:
            return None

        data = base64.b64encode(path.read_bytes()).decode("utf-8")

        return f"data:{mime};base64,{data}"

    except Exception:
        return None


def resolve_project_path(value):
    if not value:
        return None

    try:
        path = Path(str(value))

        if path.is_absolute():
            return path

        candidate = PROJECT_ROOT / path

        if candidate.exists():
            return candidate

        candidate = METADATA_DIR / path

        if candidate.exists():
            return candidate

        return None

    except Exception:
        return None


def find_exact_sample_image(sample_id):
    """
    Resolve the original handwriting image only for the requested sample.
    Never perform a global fuzzy search that could accidentally select
    another sample.
    """

    sample_id = normalize_id(sample_id)

    if not sample_id:
        return None

    # First use metadata_all.csv.
    if METADATA_ALL_FILE.exists():

        try:
            metadata = pd.read_csv(METADATA_ALL_FILE)

            id_column = None

            for col in ["sample_id", "form_id", "id"]:
                if col in metadata.columns:
                    id_column = col
                    break

            if id_column:

                rows = metadata[
                    metadata[id_column].astype(str).str.strip() == sample_id
                ]

                if not rows.empty:

                    row = rows.iloc[0]

                    for col in [
                        "image_path",
                        "path",
                        "image",
                        "file_path",
                    ]:

                        if col in metadata.columns:

                            path = resolve_project_path(row[col])

                            if path and path.exists():
                                return path

        except Exception:
            pass

    # Exact filename fallbacks only.
    possible = [
        DATA_DIR / "raw" / "IAM" / "images" / f"{sample_id}.png",
        DATA_DIR / "raw" / "IAM" / "images" / f"{sample_id}.jpg",
        DATA_DIR / "raw" / "IAM" / "images" / f"{sample_id}.jpeg",
    ]

    for path in possible:
        if path.exists():
            return path

    # Search IAM image folders only for exact filename.
    iam_images = DATA_DIR / "raw" / "IAM" / "images"

    if iam_images.exists():

        for suffix in [".png", ".jpg", ".jpeg", ".bmp"]:

            matches = list(iam_images.rglob(sample_id + suffix))

            if matches:
                return matches[0]

    return None


# ============================================================
# SAMPLE FOLDER / WORD IMAGE RESOLUTION
# ============================================================

def get_sample_explainability_folder(sample_id):
    return METADATA_DIR / f"{sample_id}_word_explainability"


def resolve_word_image(
    sample_id,
    word_id,
    word,
    rank,
    supplied_path=None,
    image_type="original",
):
    """
    Resolve an image strictly inside the requested sample's
    explainability folder.

    Priority:
    1. exact supplied path
    2. exact word_id
    3. exact rank
    4. sanitized word

    NEVER searches other sample folders.
    """

    sample_id = normalize_id(sample_id)
    word_id = normalize_id(word_id)
    word = safe_text(word, "")
    folder = get_sample_explainability_folder(sample_id)

    if not folder.exists():
        return None

    # --------------------------------------------------------
    # 1. Supplied path from word_explainability_report.csv
    # --------------------------------------------------------

    if supplied_path:
        supplied_path = str(supplied_path).strip()

        if supplied_path:

            candidate = Path(supplied_path)

            candidates = []

            if candidate.is_absolute():
                candidates.append(candidate)
            else:
                candidates.extend([
                    PROJECT_ROOT / candidate,
                    METADATA_DIR / candidate,
                    folder / candidate,
                ])

            for candidate in candidates:

                try:
                    candidate = candidate.resolve()

                    # Security/sample isolation:
                    folder_resolved = folder.resolve()

                    if (
                        candidate.exists()
                        and folder_resolved in candidate.parents
                    ):
                        return candidate

                except Exception:
                    pass

    # --------------------------------------------------------
    # File suffix
    # --------------------------------------------------------

    suffix = f"_{image_type}.png"

    # --------------------------------------------------------
    # 2. Exact word_id
    # --------------------------------------------------------

    if word_id:

        matches = list(folder.glob(f"*_{word_id}{suffix}"))

        if matches:
            return matches[0]

        matches = list(folder.glob(f"*{word_id}*{suffix}"))

        if matches:
            return matches[0]

    # --------------------------------------------------------
    # 3. Exact rank
    # --------------------------------------------------------

    try:
        rank_number = int(float(rank))

        rank_prefix = f"{rank_number:02d}_"

        matches = list(folder.glob(f"{rank_prefix}*{suffix}"))

        if matches:
            return matches[0]

    except Exception:
        pass

    # --------------------------------------------------------
    # 4. Sanitized word
    # --------------------------------------------------------

    clean_word = safe_filename(word)

    if clean_word and clean_word != "unknown":

        matches = list(folder.glob(f"*_{clean_word}{suffix}"))

        if matches:
            return matches[0]

        matches = list(folder.glob(f"*{clean_word}*{suffix}"))

        if matches:
            return matches[0]

    return None


# ============================================================
# DATA LOADING
# ============================================================

def load_csv(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return pd.DataFrame()


def load_data():

    master = load_csv(MASTER_FILE)
    anomaly = load_csv(ANOMALY_FILE)
    word_anomaly = load_csv(WORD_ANOMALY_FILE)
    word_summary = load_csv(WORD_SUMMARY_FILE)
    metadata = load_csv(METADATA_ALL_FILE)
    words_metadata = load_csv(WORDS_METADATA_FILE)

    return (
        master,
        anomaly,
        word_anomaly,
        word_summary,
        metadata,
        words_metadata,
    )


# ============================================================
# SAMPLE ROW
# ============================================================

def get_row_from_file(df, sample_id, id_columns):
    if df.empty:
        return None

    for col in id_columns:

        if col not in df.columns:
            continue

        rows = df[
            df[col].astype(str).str.strip() == sample_id
        ]

        if not rows.empty:
            return rows.iloc[0]

    return None


def get_sample_row(
    sample_id,
    master,
    anomaly,
    word_summary,
    metadata,
):
    """
    Source-of-truth strategy:

    metadata:
        sample metadata / OpenCV values / image information

    anomaly:
        form-level anomaly score

    word_summary:
        word-level aggregate statistics

    master:
        integration / percentile / combined / agreement fields
    """

    row = {}

    metadata_row = get_row_from_file(
        metadata,
        sample_id,
        ["sample_id", "form_id", "id"],
    )

    anomaly_row = get_row_from_file(
        anomaly,
        sample_id,
        ["sample_id", "form_id", "id"],
    )

    word_summary_row = get_row_from_file(
        word_summary,
        sample_id,
        ["sample_id", "form_id", "id"],
    )

    master_row = get_row_from_file(
        master,
        sample_id,
        ["sample_id", "form_id", "id"],
    )

    # Base metadata.
    if metadata_row is not None:
        row.update(metadata_row.to_dict())

    # Form anomaly fields.
    if anomaly_row is not None:
        for key, value in anomaly_row.to_dict().items():
            row[key] = value

    # Word summary fields.
    if word_summary_row is not None:
        for key, value in word_summary_row.to_dict().items():
            row[key] = value

    # Master integration is authoritative for integration fields.
    if master_row is not None:
        for key, value in master_row.to_dict().items():
            row[key] = value

    return pd.Series(row)


# ============================================================
# WORD EXPLAINABILITY
# ============================================================

def load_word_explainability(sample_id):

    folder = get_sample_explainability_folder(sample_id)

    report_file = folder / "word_explainability_report.csv"

    if not report_file.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(report_file)

        if "sample_id" in df.columns:

            df = df[
                df["sample_id"].astype(str).str.strip() == sample_id
            ]

        return df

    except Exception as exc:
        print(f"WARNING: Could not read {report_file}: {exc}")
        return pd.DataFrame()


def get_word_id_column(df):

    for col in ["word_id", "id"]:
        if col in df.columns:
            return col

    return None


def filter_word_anomaly_for_sample(word_df, sample_id):

    if word_df.empty:
        return pd.DataFrame()

    for col in ["form_id", "sample_id"]:

        if col in word_df.columns:

            result = word_df[
                word_df[col].astype(str).str.strip() == sample_id
            ].copy()

            if not result.empty:
                return result

    return pd.DataFrame(columns=word_df.columns)


# ============================================================
# WORD TEXT LOOKUP
# ============================================================

def lookup_word_text(word_id, words_metadata):

    if words_metadata.empty or not word_id:
        return None

    id_column = get_word_id_column(words_metadata)

    if not id_column:
        return None

    rows = words_metadata[
        words_metadata[id_column].astype(str).str.strip() == str(word_id).strip()
    ]

    if rows.empty:
        return None

    row = rows.iloc[0]

    for col in [
        "word",
        "transcription",
        "text",
        "label",
        "word_text",
    ]:

        if col in words_metadata.columns:

            value = safe_text(row[col], "")

            if value:
                return value

    return None


# ============================================================
# DATASET PERCENTILES
# ============================================================

def percentile_rank(value, values):

    value = safe_float(value)

    if not np.isfinite(value):
        return np.nan

    clean = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).dropna()

    if clean.empty:
        return np.nan

    return float(
        (clean <= value).mean() * 100
    )


def get_dataset_percentiles(sample_id, master, word_summary):

    result = {
        "resnet_percentile": np.nan,
        "word_percentile": np.nan,
    }

    # --------------------------------------------------------
    # ResNet percentile
    # --------------------------------------------------------

    if not master.empty:

        if "sample_id" in master.columns:

            sample_rows = master[
                master["sample_id"].astype(str).str.strip() == sample_id
            ]

        elif "form_id" in master.columns:

            sample_rows = master[
                master["form_id"].astype(str).str.strip() == sample_id
            ]

        else:
            sample_rows = pd.DataFrame()

        if not sample_rows.empty:

            row = sample_rows.iloc[0]

            resnet_value = None

            for col in [
                "resnet_anomaly_score",
                "anomaly_score_0_100",
                "anomaly_score",
            ]:

                if col in master.columns:
                    resnet_value = row[col]
                    break

            if resnet_value is not None:

                for col in [
                    "resnet_anomaly_score",
                    "anomaly_score_0_100",
                    "anomaly_score",
                ]:

                    if col in master.columns:

                        result["resnet_percentile"] = percentile_rank(
                            resnet_value,
                            master[col],
                        )

                        break

    # --------------------------------------------------------
    # Word percentile
    # --------------------------------------------------------

    if not word_summary.empty:

        sample_rows = pd.DataFrame()

        if "sample_id" in word_summary.columns:

            sample_rows = word_summary[
                word_summary["sample_id"].astype(str).str.strip() == sample_id
            ]

        elif "form_id" in word_summary.columns:

            sample_rows = word_summary[
                word_summary["form_id"].astype(str).str.strip() == sample_id
            ]

        if not sample_rows.empty:

            row = sample_rows.iloc[0]

            word_value = None

            for col in [
                "mean_word_anomaly",
                "mean_anomaly",
                "word_anomaly_mean",
            ]:

                if col in word_summary.columns:
                    word_value = row[col]
                    break

            if word_value is not None:

                for col in [
                    "mean_word_anomaly",
                    "mean_anomaly",
                    "word_anomaly_mean",
                ]:

                    if col in word_summary.columns:

                        result["word_percentile"] = percentile_rank(
                            word_value,
                            word_summary[col],
                        )

                        break

    return result


# ============================================================
# AGREEMENT
# ============================================================

def get_agreement(row, percentiles):

    # Prefer the official master-analysis category.
    for col in [
        "anomaly_agreement",
        "agreement_category",
        "agreement",
    ]:

        if col in row.index:

            value = safe_text(row.get(col), "")

            if value:
                return value

    rp = percentiles["resnet_percentile"]
    wp = percentiles["word_percentile"]

    if not np.isfinite(rp) or not np.isfinite(wp):
        return "Not available"

    if rp >= 75 and wp >= 75:
        return "High in both"

    if rp >= 75 and wp < 75:
        return "High ResNet only"

    if rp < 75 and wp >= 75:
        return "High word anomaly only"

    return "Low/moderate in both"


# ============================================================
# DOMINANT WORD FEATURES
# ============================================================

def get_top_words(
    sample_id,
    word_df,
    explanation_df,
    words_metadata,
    number=10,
):

    word_df = filter_word_anomaly_for_sample(
        word_df,
        sample_id,
    )

    if word_df.empty:
        return pd.DataFrame()

    if "word_anomaly_score" not in word_df.columns:
        return pd.DataFrame()

    word_df = word_df.copy()

    word_df["word_anomaly_score"] = pd.to_numeric(
        word_df["word_anomaly_score"],
        errors="coerce",
    )

    word_df = word_df.dropna(
        subset=["word_anomaly_score"]
    )

    word_df = word_df.sort_values(
        "word_anomaly_score",
        ascending=False,
    ).head(number)

    word_id_col = get_word_id_column(word_df)

    if word_id_col:

        word_df["_join_word_id"] = (
            word_df[word_id_col]
            .astype(str)
            .str.strip()
        )

    else:

        word_df["_join_word_id"] = ""

    # --------------------------------------------------------
    # Join explainability only from this sample.
    # --------------------------------------------------------

    if not explanation_df.empty:

        explanation_df = explanation_df.copy()

        explanation_id_col = get_word_id_column(
            explanation_df
        )

        if explanation_id_col:

            explanation_df["_join_word_id"] = (
                explanation_df[explanation_id_col]
                .astype(str)
                .str.strip()
            )

            keep_cols = [
                "_join_word_id",
                "rank",
                "word",
                "original_crop",
                "gradcam_image",
                "comparison_image",
                "high_activation_percentage",
                "explanation",
            ]

            keep_cols = [
                col for col in keep_cols
                if col in explanation_df.columns
            ]

            explanation_small = explanation_df[
                keep_cols
            ].drop_duplicates(
                "_join_word_id"
            )

            word_df = word_df.merge(
                explanation_small,
                on="_join_word_id",
                how="left",
                suffixes=("", "_explanation"),
            )

    # --------------------------------------------------------
    # Retrieve actual word text.
    # --------------------------------------------------------

    words = []

    for _, row in word_df.iterrows():

        word = None

        if "word" in row.index:
            word = safe_text(row["word"], "")

        if not word and "word_explanation" in row.index:
            word = safe_text(row["word_explanation"], "")

        if not word:

            word_id = safe_text(
                row.get("_join_word_id"),
                "",
            )

            word = lookup_word_text(
                word_id,
                words_metadata,
            )

        words.append(
            word if word else "Unknown"
        )

    word_df["resolved_word"] = words

    return word_df


def dominant_word_features(top_words):

    if top_words.empty:
        return []

    feature_counts = {}

    for _, row in top_words.iterrows():

        feature = safe_text(
            row.get("main_anomaly_feature"),
            "",
        )

        if not feature:
            continue

        feature_counts[feature] = (
            feature_counts.get(feature, 0) + 1
        )

    return sorted(
        feature_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )


# ============================================================
# FEATURE INTERPRETATION
# ============================================================

def interpret_sample_feature(feature, value):

    value = safe_float(value)

    if not np.isfinite(value):
        return "Measurement unavailable."

    explanations = {

        "skew":
            "Measures the estimated orientation/skew of the writing layout.",

        "baseline_deviation":
            "Measures variation in the estimated writing baseline.",

        "word_spacing_cv":
            "Measures relative variability in spacing between detected words.",

        "character_height_cv":
            "Measures relative variability in character height.",

        "average_word_height":
            "Measures the average vertical size of detected word regions.",

        "average_word_width":
            "Measures the average horizontal size of detected word regions.",

        "stroke_density":
            "Measures the proportion of the writing region occupied by detected ink.",

        "slant_angle":
            "Measures the estimated directional inclination of handwriting strokes.",

        "writing_area":
            "Measures the proportion of the normalized form occupied by writing.",

        "connected_component_count":
            "Counts connected ink components detected in the complete sample.",

        "num_lines":
            "Number of handwriting lines detected in the form.",

        "num_words":
            "Number of word regions detected during form-level segmentation.",

        "num_components":
            "Number of components retained by the segmentation pipeline.",

        "segmentation_error_count":
            "Number of segmentation events marked as errors.",
    }

    return explanations.get(
        feature,
        "Handwriting measurement extracted from the sample.",
    )


# ============================================================
# AUTOMATIC SAMPLE FINDINGS
# ============================================================

def build_sample_findings(
    sample_id,
    row,
    top_words,
    percentiles,
    agreement,
):

    resnet = safe_float(
        row.get("resnet_anomaly_score")
    )

    if not np.isfinite(resnet):
        resnet = safe_float(
            row.get("anomaly_score_0_100")
        )

    mean_word = safe_float(
        row.get("mean_word_anomaly")
    )

    max_word = safe_float(
        row.get("maximum_word_anomaly")
    )

    top5 = safe_float(
        row.get("top_5_mean_anomaly")
    )

    rp = percentiles["resnet_percentile"]
    wp = percentiles["word_percentile"]

    # --------------------------------------------------------
    # Dominant word features.
    # --------------------------------------------------------

    feature_counts = dominant_word_features(
        top_words
    )

    dominant_feature = (
        feature_counts[0][0]
        if feature_counts
        else None
    )

    dominant_count = (
        feature_counts[0][1]
        if feature_counts
        else 0
    )

    # --------------------------------------------------------
    # Highest Z-score.
    # --------------------------------------------------------

    highest_z = np.nan
    highest_z_feature = None

    if not top_words.empty:

        for _, word in top_words.iterrows():

            z = safe_float(
                word.get("main_feature_z")
            )

            if np.isfinite(z):

                if not np.isfinite(highest_z) or abs(z) > abs(highest_z):

                    highest_z = z

                    highest_z_feature = safe_text(
                        word.get(
                            "main_anomaly_feature"
                        ),
                        "",
                    )

    # --------------------------------------------------------
    # Agreement narrative.
    # --------------------------------------------------------

    if agreement == "High in both":

        headline = (
            "GLOBAL + LOCAL ANOMALY EVIDENCE"
        )

        finding = (
            f"Sample {sample_id} shows elevated anomaly evidence "
            "at both the whole-form representation level and the "
            "individual-word level."
        )

        interpretation = (
            "The form-level ResNet50 representation and multiple "
            "word-level measurements independently identify this "
            "sample as relatively unusual compared with their "
            "respective reference distributions."
        )

    elif agreement == "High ResNet only":

        headline = (
            "FORM-LEVEL ANOMALY DOMINATES"
        )

        finding = (
            f"Sample {sample_id} is substantially more unusual "
            "at the whole-form representation level than at the "
            "individual-word level."
        )

        interpretation = (
            "The ResNet50 representation captures strong global "
            "visual deviation, while the word-level measurements "
            "show comparatively moderate deviations. This means "
            "the strongest evidence for this sample is distributed "
            "across broader handwriting appearance rather than "
            "being concentrated in highly anomalous individual words."
        )

    elif agreement == "High word anomaly only":

        headline = (
            "WORD-LEVEL ANOMALY DOMINATES"
        )

        finding = (
            f"Sample {sample_id} has relatively moderate whole-form "
            "representation anomaly but elevated anomalies in "
            "individual handwriting regions."
        )

        interpretation = (
            "The unusual characteristics appear to be more localized "
            "to particular word regions than strongly expressed across "
            "the complete handwriting form."
        )

    else:

        headline = (
            "LOWER ANOMALY AT BOTH LEVELS"
        )

        finding = (
            f"Sample {sample_id} does not show strong anomaly evidence "
            "at either the whole-form or aggregate word level."
        )

        interpretation = (
            "The sample is comparatively close to the reference "
            "distribution used by the research pipeline."
        )

    # --------------------------------------------------------
    # Word-level pattern.
    # --------------------------------------------------------

    if dominant_feature:

        feature_label = dict(
            WORD_FEATURES
        ).get(
            dominant_feature,
            dominant_feature.replace("_", " ").title(),
        )

        word_pattern = (
            f"The dominant word-level anomaly feature among the "
            f"top-ranked words is <strong>{html_escape(feature_label)}</strong>, "
            f"appearing in {dominant_count} of the top {len(top_words)} "
            "anomalous words."
        )

    else:

        word_pattern = (
            "A dominant word-level anomaly feature could not be "
            "determined from the available records."
        )

    # --------------------------------------------------------
    # Z-score pattern.
    # --------------------------------------------------------

    if np.isfinite(highest_z):

        z_label = (
            dict(WORD_FEATURES).get(
                highest_z_feature,
                safe_text(
                    highest_z_feature,
                    "word-level feature",
                ).replace("_", " ").title(),
            )
        )

        z_pattern = (
            f"The strongest absolute robust Z-score among the "
            f"top-ranked words is <strong>{format_number(highest_z, 2)}</strong> "
            f"for {html_escape(z_label)}."
        )

    else:

        z_pattern = (
            "A robust Z-score was not available for the selected "
            "word-level records."
        )

    # --------------------------------------------------------
    # Percentile evidence.
    # --------------------------------------------------------

    percentile_text = []

    if np.isfinite(rp):

        percentile_text.append(
            f"the form-level ResNet50 score is at approximately "
            f"the {format_percent(rp, 1)} dataset percentile"
        )

    if np.isfinite(wp):

        percentile_text.append(
            f"the mean word anomaly is at approximately "
            f"the {format_percent(wp, 1)} dataset percentile"
        )

    if percentile_text:

        relative_position = (
            " and ".join(percentile_text) + "."
        )

    else:

        relative_position = (
            "dataset percentile information is not available."
        )

    # --------------------------------------------------------
    # Sample-specific conclusion.
    # --------------------------------------------------------

    if agreement == "High in both":

        what_it_tells = (
            f"The evidence for {sample_id} is distributed across "
            "both global and local handwriting characteristics. "
            "The whole-form representation identifies the sample "
            "as unusual while several individual words also show "
            "elevated deviations. "
            f"{word_pattern} "
            f"{z_pattern} "
            "Taken together, this makes the sample a strong example "
            "of agreement between the two anomaly-detection views."
        )

    elif agreement == "High ResNet only":

        what_it_tells = (
            f"The main signal for {sample_id} comes from the complete "
            "handwriting representation rather than extreme individual "
            "word anomalies. "
            f"{word_pattern} "
            f"{z_pattern} "
            "The word-level evidence is present but comparatively "
            "weaker, which is why the two pipelines do not produce "
            "a High in both result. This sample demonstrates the "
            "value of separating global handwriting representation "
            "from local word-level analysis."
        )

    elif agreement == "High word anomaly only":

        what_it_tells = (
            f"The main signal for {sample_id} is localized to individual "
            "word regions rather than strongly expressed across the "
            "complete handwriting form. "
            f"{word_pattern} "
            f"{z_pattern} "
            "This indicates that local analysis contributes information "
            "that may be less visible in the global representation."
        )

    else:

        what_it_tells = (
            f"The available evidence for {sample_id} is relatively "
            "moderate at both analysis levels. "
            f"{word_pattern} "
            f"{z_pattern} "
            "The sample therefore provides comparatively limited "
            "evidence of strong deviation from the reference "
            "distribution used by this research pipeline."
        )

    return {
        "headline": headline,
        "finding": finding,
        "interpretation": interpretation,
        "word_pattern": word_pattern,
        "z_pattern": z_pattern,
        "relative_position": relative_position,
        "what_it_tells": what_it_tells,
        "dominant_feature": dominant_feature,
        "dominant_count": dominant_count,
        "highest_z": highest_z,
        "highest_z_feature": highest_z_feature,
    }


# ============================================================
# HTML COMPONENTS
# ============================================================

def image_html(path, alt, class_name="report-image"):

    encoded = image_to_base64(path)

    if not encoded:

        return f"""
        <div class="missing-image">
            <strong>{html_escape(alt)}</strong>
            <br>
            Visual not available for this exact sample/word.
        </div>
        """

    return f"""
    <div class="image-container">
        <img
            src="{encoded}"
            alt="{html_escape(alt)}"
            class="{class_name}"
        >
    </div>
    """


def create_sample_specific_section(
    sample_id,
    findings,
    agreement,
):

    return f"""
    <section>
        <h2>1. Sample-Specific Finding</h2>

        <div class="finding-hero">
            <div class="finding-badge">
                {html_escape(findings["headline"])}
            </div>

            <h3>{html_escape(sample_id)}</h3>

            <p>
                {findings["finding"]}
            </p>

            <p>
                {findings["interpretation"]}
            </p>

            <div class="evidence-grid">

                <div class="evidence-card">
                    <span>Agreement</span>
                    <strong>{html_escape(agreement)}</strong>
                </div>

                <div class="evidence-card">
                    <span>Dominant Word Feature</span>
                    <strong>
                        {html_escape(
                            findings["dominant_feature"] or "N/A"
                        ).replace("_", " ")}
                    </strong>
                </div>

                <div class="evidence-card">
                    <span>Highest Word Z-score</span>
                    <strong>
                        {format_number(findings["highest_z"], 2)}
                    </strong>
                </div>

            </div>

            <h3>What this sample tells us</h3>

            <p>
                {findings["what_it_tells"]}
            </p>

            <div class="note">
                <strong>Important:</strong>
                These findings describe statistical and representational
                deviations from the reference distributions. They do not
                establish dysgraphia or any other clinical condition.
            </div>
        </div>
    </section>
    """


def create_summary_cards(
    resnet,
    mean_word,
    max_word,
    top5,
    combined,
    agreement,
):

    return f"""
    <div class="summary-grid">

        <div class="metric-card">
            <span>Form-Level ResNet50</span>
            <strong>{format_number(resnet, 2)}</strong>
        </div>

        <div class="metric-card">
            <span>Mean Word Anomaly</span>
            <strong>{format_number(mean_word, 3)}</strong>
        </div>

        <div class="metric-card">
            <span>Maximum Word Anomaly</span>
            <strong>{format_number(max_word, 3)}</strong>
        </div>

        <div class="metric-card">
            <span>Top-5 Mean Word Anomaly</span>
            <strong>{format_number(top5, 3)}</strong>
        </div>

        <div class="metric-card">
            <span>Combined Research Score</span>
            <strong>{format_number(combined, 2)}</strong>
        </div>

        <div class="metric-card">
            <span>Agreement</span>
            <strong>{html_escape(agreement)}</strong>
        </div>

    </div>
    """


def create_sample_features_table(row):

    rows = []

    for key, label in SAMPLE_FEATURES:

        value = row.get(key)

        rows.append(
            f"""
            <tr>
                <td>{html_escape(label)}</td>
                <td>{format_number(value, 4)}</td>
                <td>
                    {html_escape(
                        interpret_sample_feature(key, value)
                    )}
                </td>
            </tr>
            """
        )

    return f"""
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>What it measures</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def create_word_feature_table(row):

    rows = []

    for feature, label in WORD_FEATURES:

        value = row.get(feature)

        z_column = f"{feature}_z"

        z = row.get(z_column)

        if not np.isfinite(safe_float(z)):

            # Some pipelines store the z-score in a generic
            # column or only the main feature Z-score.
            main_feature = safe_text(
                row.get("main_anomaly_feature"),
                "",
            )

            if main_feature == feature:
                z = row.get("main_feature_z")

        rows.append(
            f"""
            <tr>
                <td>{html_escape(label)}</td>
                <td>{format_number(value, 4)}</td>
                <td>{format_number(z, 3)}</td>
            </tr>
            """
        )

    return f"""
    <table class="small-table">
        <thead>
            <tr>
                <th>Feature</th>
                <th>Value</th>
                <th>Robust Z-score</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def create_word_section(
    sample_id,
    row,
    rank,
):

    word_id = safe_text(
        row.get("word_id"),
        "",
    )

    word = safe_text(
        row.get("resolved_word"),
        "Unknown",
    )

    score = safe_float(
        row.get("word_anomaly_score")
    )

    main_feature = safe_text(
        row.get("main_anomaly_feature"),
        "N/A",
    )

    z = safe_float(
        row.get("main_feature_z")
    )

    activation = safe_float(
        row.get("high_activation_percentage")
    )

    explanation = safe_text(
        row.get("explanation"),
        "",
    )

    if not explanation:

        feature_label = dict(
            WORD_FEATURES
        ).get(
            main_feature,
            main_feature.replace("_", " ").title(),
        )

        explanation = (
            f"The word has a word anomaly score of "
            f"{format_number(score, 2)}. The primary contributing "
            f"feature is {feature_label}, with a robust Z-score of "
            f"{format_number(z, 2)}. The Grad-CAM highlights regions "
            "that contribute strongly to the ResNet50 handwriting "
            "representation. This visualization is model evidence "
            "of unusual representation patterns and is not a "
            "clinical diagnosis."
        )

    original_path = resolve_word_image(
        sample_id,
        word_id,
        word,
        rank,
        row.get("original_crop"),
        "original",
    )

    gradcam_path = resolve_word_image(
        sample_id,
        word_id,
        word,
        rank,
        row.get("gradcam_image"),
        "gradcam",
    )

    comparison_path = resolve_word_image(
        sample_id,
        word_id,
        word,
        rank,
        row.get("comparison_image"),
        "comparison",
    )

    return f"""
    <div class="word-card">

        <h3>
            #{rank} — {html_escape(word)}
        </h3>

        <div class="word-meta">
            <strong>Anomaly Score:</strong>
            {format_number(score, 3)}
            &nbsp;&nbsp;|&nbsp;&nbsp;

            <strong>Word ID:</strong>
            {html_escape(word_id)}
            &nbsp;&nbsp;|&nbsp;&nbsp;

            <strong>Main Feature:</strong>
            {html_escape(main_feature)}
            &nbsp;&nbsp;|&nbsp;&nbsp;

            <strong>Z-score:</strong>
            {format_number(z, 3)}
        </div>

        <div class="word-images">

            <div>
                <h4>Original Word</h4>
                {image_html(
                    original_path,
                    f"Original word {word}"
                )}
            </div>

            <div>
                <h4>ResNet50 Grad-CAM</h4>
                {image_html(
                    gradcam_path,
                    f"Grad-CAM for {word}"
                )}
            </div>

            <div>
                <h4>Original + Grad-CAM</h4>
                {image_html(
                    comparison_path,
                    f"Comparison for {word}"
                )}
            </div>

        </div>

        <h4>Word-Level Feature Measurements</h4>

        {create_word_feature_table(row)}

        <p>
            <strong>High-activation region:</strong>
            {format_percent(activation, 2)}
        </p>

        <p>
            <strong>Interpretation:</strong>
            {html_escape(explanation)}
        </p>

    </div>
    """


# ============================================================
# HTML GENERATION
# ============================================================

def generate_html(
    sample_id,
    row,
    master,
    word_anomaly,
    word_summary,
    metadata,
    words_metadata,
):

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    resnet = safe_float(
        row.get("resnet_anomaly_score")
    )

    if not np.isfinite(resnet):
        resnet = safe_float(
            row.get("anomaly_score_0_100")
        )

    mean_word = safe_float(
        row.get("mean_word_anomaly")
    )

    max_word = safe_float(
        row.get("maximum_word_anomaly")
    )

    top5 = safe_float(
        row.get("top_5_mean_anomaly")
    )

    combined = safe_float(
        row.get("combined_anomaly_score")
    )

    if not np.isfinite(combined):

        rp = percentile_rank(
            resnet,
            master.get("resnet_anomaly_score", []),
        )

        wp = percentile_rank(
            mean_word,
            word_summary.get("mean_word_anomaly", []),
        )

        if np.isfinite(rp) and np.isfinite(wp):
            combined = 0.5 * rp + 0.5 * wp

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    percentiles = get_dataset_percentiles(
        sample_id,
        master,
        word_summary,
    )

    agreement = get_agreement(
        row,
        percentiles,
    )

    # --------------------------------------------------------
    # Words
    # --------------------------------------------------------

    explanation_df = load_word_explainability(
        sample_id
    )

    top_words = get_top_words(
        sample_id,
        word_anomaly,
        explanation_df,
        words_metadata,
        number=10,
    )

    # --------------------------------------------------------
    # Findings
    # --------------------------------------------------------

    findings = build_sample_findings(
        sample_id,
        row,
        top_words,
        percentiles,
        agreement,
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original_image = find_exact_sample_image(
        sample_id
    )

    # --------------------------------------------------------
    # Word count
    # --------------------------------------------------------

    form_word_count = safe_float(
        row.get("num_words")
    )

    valid_word_count = len(
        filter_word_anomaly_for_sample(
            word_anomaly,
            sample_id,
        )
    )

    explanation_count = len(
        explanation_df
    )

    # --------------------------------------------------------
    # Word feature frequency
    # --------------------------------------------------------

    feature_counts = dominant_word_features(
        top_words
    )

    feature_rows = []

    for feature, count in feature_counts:

        label = dict(
            WORD_FEATURES
        ).get(
            feature,
            feature.replace("_", " ").title(),
        )

        feature_rows.append(
            f"""
            <tr>
                <td>{html_escape(label)}</td>
                <td>{count}</td>
            </tr>
            """
        )

    # --------------------------------------------------------
    # Top words
    # --------------------------------------------------------

    word_sections = []

    for rank, (_, word_row) in enumerate(
        top_words.iterrows(),
        start=1,
    ):

        word_sections.append(
            create_word_section(
                sample_id,
                word_row,
                rank,
            )
        )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    original_section = image_html(
        original_image,
        f"Original handwriting sample {sample_id}",
        "sample-image",
    )

    # --------------------------------------------------------
    # Dataset chart
    # --------------------------------------------------------

    chart_section = image_html(
        COMPARISON_CHART,
        "Dataset-level ResNet50 versus word anomaly comparison",
    )

    # --------------------------------------------------------
    # Missing explainability message
    # --------------------------------------------------------

    if explanation_df.empty:

        explainability_status = """
        <div class="warning">
            Sample-specific word explainability data was not found.
            Word-level anomaly results are still shown, but Grad-CAM
            images have not been generated for this sample.
        </div>
        """

    else:

        explainability_status = f"""
        <div class="success">
            Sample-specific word explainability data found.
            {explanation_count} explainability records are associated
            with sample <strong>{html_escape(sample_id)}</strong>.
        </div>
        """

    # --------------------------------------------------------
    # Percentile display
    # --------------------------------------------------------

    rp = percentiles["resnet_percentile"]
    wp = percentiles["word_percentile"]

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    return f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<title>
Explainable Handwriting Anomaly Report - {html_escape(sample_id)}
</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 0;
    background: #f4f7fb;
    color: #1f2937;
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: auto;
    padding: 30px;
}}

header {{
    background: #111827;
    color: white;
    padding: 35px;
    border-radius: 14px;
    margin-bottom: 25px;
}}

h1 {{
    margin-top: 0;
}}

h2 {{
    margin-top: 0;
    color: #111827;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 10px;
}}

h3 {{
    color: #1f2937;
}}

section {{
    background: white;
    padding: 28px;
    margin-bottom: 25px;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
}}

.summary-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}}

.metric-card {{
    background: #f8fafc;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}}

.metric-card span {{
    display: block;
    color: #64748b;
    font-size: 13px;
    margin-bottom: 7px;
}}

.metric-card strong {{
    font-size: 25px;
}}

.finding-hero {{
    background: #f8fafc;
    padding: 25px;
    border-radius: 12px;
    border-left: 6px solid #2563eb;
}}

.finding-badge {{
    display: inline-block;
    padding: 8px 14px;
    border-radius: 20px;
    background: #dbeafe;
    color: #1e40af;
    font-weight: bold;
    margin-bottom: 12px;
}}

.evidence-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin: 20px 0;
}}

.evidence-card {{
    background: white;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}}

.evidence-card span {{
    display: block;
    color: #64748b;
    font-size: 13px;
}}

.evidence-card strong {{
    display: block;
    margin-top: 6px;
    font-size: 18px;
}}

.sample-image {{
    max-width: 90%;
    max-height: 800px;
    display: block;
    margin: auto;
}}

.image-container {{
    background: #f8fafc;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}}

.report-image {{
    max-width: 100%;
    max-height: 450px;
    object-fit: contain;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
}}

th {{
    background: #e5e7eb;
    text-align: left;
}}

th, td {{
    padding: 10px;
    border: 1px solid #d1d5db;
}}

.small-table {{
    font-size: 13px;
}}

.word-card {{
    border: 1px solid #dbe2ea;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 25px;
    background: #fafcff;
}}

.word-meta {{
    color: #475569;
    margin-bottom: 18px;
}}

.word-images {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}}

.missing-image {{
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: #f1f5f9;
    color: #64748b;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}}

.warning {{
    background: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 15px;
    border-radius: 8px;
}}

.success {{
    background: #f0fdf4;
    border-left: 5px solid #22c55e;
    padding: 15px;
    border-radius: 8px;
}}

.note {{
    background: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
}}

.legend {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin: 15px 0;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
}}

.legend-box {{
    width: 22px;
    height: 22px;
    border-radius: 4px;
}}

footer {{
    text-align: center;
    color: #64748b;
    padding: 30px;
}}

@media(max-width: 850px) {{

    .summary-grid,
    .evidence-grid,
    .word-images {{
        grid-template-columns: 1fr;
    }}

}}

</style>

</head>

<body>

<div class="container">

<header>

<h1>Explainable Handwriting Anomaly Detection Report</h1>

<p>
<strong>Sample ID:</strong>
{html_escape(sample_id)}
</p>

<p>
ResNet50 + OpenCV + Word-Level Anomaly Analysis + Grad-CAM
</p>

</header>


<!-- ======================================================
     SECTION 1
     ====================================================== -->

{create_sample_specific_section(
    sample_id,
    findings,
    agreement,
)}


<!-- ======================================================
     SECTION 2
     ====================================================== -->

<section>

<h2>2. Executive Summary</h2>

{create_summary_cards(
    resnet,
    mean_word,
    max_word,
    top5,
    combined,
    agreement,
)}

<div class="note">

<strong>Interpretation:</strong>

The anomaly scores describe how unusual the handwriting appears
relative to the reference distribution used by the research pipeline.

They are not dysgraphia probabilities, clinical severity scores,
confidence scores, or medical diagnoses.

</div>

</section>


<!-- ======================================================
     SECTION 3
     ====================================================== -->

<section>

<h2>3. Original Handwriting Sample</h2>

<p>
Original handwriting sample:
<strong>{html_escape(sample_id)}</strong>
</p>

{original_section}

</section>


<!-- ======================================================
     SECTION 4
     ====================================================== -->

<section>

<h2>4. Sample-Level OpenCV Measurements</h2>

<p>
These measurements are extracted from the complete handwriting
form. They are separate from the word-level anomaly features.
</p>

{create_sample_features_table(row)}

</section>


<!-- ======================================================
     SECTION 5
     ====================================================== -->

<section>

<h2>5. Feature and Model Architecture</h2>

<div style="
background:#f8fafc;
padding:25px;
border-radius:12px;
text-align:center;
font-weight:bold;
">

<p>Handwriting Image</p>

<p>↓</p>

<p>
ResNet50<br>
2048-D
</p>

<p>+</p>

<p>
9 Selected<br>
Sample-Level / Form-Level<br>
OpenCV Features
</p>

<p>↓</p>

<p>
OpenCV Encoder<br>
64-D
</p>

<p>↓</p>

<p>
Fusion<br>
2112-D
</p>

<p>↓</p>

<p>
Feature Attention
</p>

<p>↓</p>

<p>
256-D Shared Representation
</p>

<p>↓</p>

<p>
Anomaly Score
</p>

</div>

<p>
The ResNet50 branch uses ImageNet-pretrained visual representations,
while the OpenCV branch contributes handcrafted handwriting
measurements. Their representations are fused for sample-level
anomaly detection.
</p>

</section>


<!-- ======================================================
     SECTION 6
     ====================================================== -->

<section>

<h2>6. Word-Level Anomaly Analysis</h2>

<p>
The form-level pipeline detected approximately
<strong>{format_number(form_word_count, 0)}</strong> words,
while the word-anomaly pipeline analyzed
<strong>{valid_word_count}</strong> valid handwriting words.
These counts can differ because the word-level pipeline filters
invalid, punctuation, and unsuitable word regions before anomaly
scoring.
</p>

<p>
The word-level pipeline evaluates individual handwriting regions
using robust feature-based anomaly scoring. Higher values indicate
greater deviation from the reference word distribution.
</p>

{explainability_status}

<h3>Dominant Word-Level Pattern</h3>

<p>
{findings["word_pattern"]}
</p>

<p>
{findings["z_pattern"]}
</p>

<table>

<thead>
<tr>
<th>Recurring Feature</th>
<th>Occurrences in Top 10</th>
</tr>
</thead>

<tbody>
{''.join(feature_rows) if feature_rows else
'<tr><td colspan="2">No dominant feature available.</td></tr>'}
</tbody>

</table>

</section>


<!-- ======================================================
     SECTION 7
     ====================================================== -->

<section>

<h2>7. Top 10 Anomalous Words</h2>

<p>
The following words are ranked by their word-level anomaly score
within the selected sample.
</p>

{''.join(word_sections) if word_sections else
'<p>No word-level anomaly records were available for this sample.</p>'}

</section>


<!-- ======================================================
     SECTION 8
     ====================================================== -->

<section>

<h2>8. ResNet50 and Grad-CAM Explanation</h2>

<p>
Grad-CAM highlights image regions that contribute strongly to the
ResNet50 handwriting representation. It provides visual localization
of influential regions; it should not be interpreted as a direct
gradient explanation of the final anomaly score.
</p>

<h3>Heatmap Interpretation</h3>

<div class="legend">

<div class="legend-item">
<div class="legend-box" style="background:#0000ff;"></div>
Blue → lower activation
</div>

<div class="legend-item">
<div class="legend-box" style="background:#00ff00;"></div>
Green → moderate activation
</div>

<div class="legend-item">
<div class="legend-box" style="background:#ffff00;"></div>
Yellow → stronger activation
</div>

<div class="legend-item">
<div class="legend-box" style="background:#ff0000;"></div>
Red → highest activation
</div>

</div>

<p>
The visualizations shown in this report are restricted to the exact
sample and word IDs requested for <strong>{html_escape(sample_id)}</strong>.
</p>

</section>


<!-- ======================================================
     SECTION 9
     ====================================================== -->

<section>

<h2>9. Global vs Local Evidence</h2>

<p>
The dataset-level comparison below provides context for the relationship
between form-level ResNet50 anomaly scores and aggregate word-level
anomaly measurements.
</p>

{chart_section}

<p>
<strong>ResNet percentile:</strong>
{format_percent(rp, 1)}
</p>

<p>
<strong>Word anomaly percentile:</strong>
{format_percent(wp, 1)}
</p>

<p>
<strong>Agreement category:</strong>
{html_escape(agreement)}
</p>

<p>
{findings["relative_position"]}
</p>

<div class="note">

<strong>Combined research score:</strong>

When available, this score is calculated using a 50/50
percentile-based research heuristic combining the form-level
ResNet anomaly percentile and word-level anomaly percentile.

It is not a probability, confidence score, clinical severity score,
or diagnostic measure.

</div>

</section>


<!-- ======================================================
     SECTION 10
     ====================================================== -->

<section>

<h2>10. Sample-Specific Findings and Interpretation</h2>

<h3>What is unusual?</h3>

<p>
{findings["finding"]}
</p>

<h3>Where is the evidence?</h3>

<p>
{findings["word_pattern"]}
</p>

<h3>How strong is the local evidence?</h3>

<p>
{findings["z_pattern"]}
</p>

<h3>How do the two pipelines agree?</h3>

<p>
The agreement category for this sample is
<strong>{html_escape(agreement)}</strong>.
This category is dataset-relative and describes agreement between
two anomaly-detection approaches rather than a clinical condition.
</p>

<h3>What this sample tells us</h3>

<p>
{findings["what_it_tells"]}
</p>

</section>


<!-- ======================================================
     SECTION 11
     ====================================================== -->

<section>

<h2>11. Limitations</h2>

<ul>

<li>
The IAM dataset primarily provides adult handwriting and does not
provide reliable clinical dysgraphia labels for this pipeline.
</li>

<li>
Anomaly scores indicate deviation from a reference distribution and
are not diagnostic probabilities.
</li>

<li>
Word-level counts can differ from form-level detected word counts
because of filtering and segmentation rules.
</li>

<li>
Grad-CAM provides visual localization of influential ResNet50
representation regions rather than a direct explanation of the
final anomaly score.
</li>

<li>
The combined score is a research heuristic and requires validation
on clinically labelled handwriting data before any diagnostic
interpretation.
</li>

</ul>

</section>


<!-- ======================================================
     SECTION 12
     ====================================================== -->

<section>

<h2>12. Sample-Specific Conclusion</h2>

<p>

For <strong>{html_escape(sample_id)}</strong>,
the evidence indicates:

</p>

<ul>

<li>
<strong>Global evidence:</strong>
{format_number(resnet, 2)}
</li>

<li>
<strong>Mean local word anomaly:</strong>
{format_number(mean_word, 3)}
</li>

<li>
<strong>Strongest word anomaly:</strong>
{format_number(max_word, 3)}
</li>

<li>
<strong>Agreement:</strong>
{html_escape(agreement)}
</li>

<li>
<strong>Dominant local pattern:</strong>
{html_escape(
    findings["dominant_feature"] or "Not available"
)}
</li>

</ul>

<p>
{findings["what_it_tells"]}
</p>

<p>
This report therefore provides a sample-specific combination of
global handwriting representation, handcrafted measurements,
local word-level anomaly evidence, and visual explanation.
The system is intended as an explainable handwriting anomaly-analysis
framework and research tool, not as a standalone clinical diagnostic
system.
</p>

</section>


<footer>

<strong>
Explainable Handwriting Anomaly Detection using ResNet50 and
OpenCV Features
</strong>

<br>

ResNet50 + OpenCV Features + Feature Fusion + Attention +
Anomaly Detection + Word-Level Analysis + Grad-CAM

</footer>

</div>

</body>

</html>
"""


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate a sample-specific handwriting anomaly report."
    )

    parser.add_argument(
        "--sample",
        "--sample_id",
        dest="sample_id",
        required=True,
        help="IAM sample ID, e.g. a01-063",
    )

    args = parser.parse_args()

    sample_id = args.sample_id.strip()

    print("=" * 70)
    print("EXPLAINABLE HANDWRITING ANOMALY REPORT")
    print("=" * 70)
    print(f"Sample ID: {sample_id}")
    print()

    (
        master,
        anomaly,
        word_anomaly,
        word_summary,
        metadata,
        words_metadata,
    ) = load_data()

    # --------------------------------------------------------
    # Verify sample exists.
    # --------------------------------------------------------

    sample_row = get_sample_row(
        sample_id,
        master,
        anomaly,
        word_summary,
        metadata,
    )

    if sample_row.empty:

        print(
            f"ERROR: Sample '{sample_id}' was not found "
            "in the available metadata/anomaly files."
        )

        return

    # --------------------------------------------------------
    # Generate HTML.
    # --------------------------------------------------------

    html_content = generate_html(
        sample_id,
        sample_row,
        master,
        word_anomaly,
        word_summary,
        metadata,
        words_metadata,
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR /
        f"{safe_filename(sample_id)}_final_report.html"
    )

    output_file.write_text(
        html_content,
        encoding="utf-8",
    )

    print()
    print("DONE.")
    print()
    print(f"Report generated:")
    print(output_file)
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()