from pathlib import Path
import argparse
import base64
import html
import re
import subprocess
import sys

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

# Existing comparison chart.
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

FORM_LOF_FEATURES = [
    ("skew", "Skew"),
    ("baseline_deviation", "Baseline Deviation"),
    ("word_spacing_cv", "Word Spacing CV"),
    ("character_height_cv", "Character Height CV"),
    ("average_word_height", "Average Word Height"),
    ("average_word_width", "Average Word Width"),
    ("stroke_density", "Stroke Density"),
    ("slant_angle", "Slant Angle"),
    ("writing_area", "Writing Area"),
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


def feature_label(feature, feature_list):
    feature = safe_text(feature, "")

    if not feature:
        return "N/A"

    mapping = dict(feature_list)

    return mapping.get(
        feature,
        feature.replace("_", " ").title(),
    )


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

        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower())

        if not mime:
            return None

        data = base64.b64encode(
            path.read_bytes()
        ).decode("utf-8")

        return f"data:{mime};base64,{data}"

    except Exception:
        return None


def resolve_project_path(value):
    if not value:
        return None

    try:
        path = Path(str(value))

        if path.is_absolute():
            return path if path.exists() else None

        candidates = [
            PROJECT_ROOT / path,
            METADATA_DIR / path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

    except Exception:
        pass

    return None


def find_exact_sample_image(sample_id):

    sample_id = normalize_id(sample_id)

    if not sample_id:
        return None

    # --------------------------------------------------------
    # Metadata lookup
    # --------------------------------------------------------

    if METADATA_ALL_FILE.exists():

        try:

            metadata = pd.read_csv(
                METADATA_ALL_FILE
            )

            id_column = None

            for col in [
                "sample_id",
                "form_id",
                "id",
            ]:

                if col in metadata.columns:
                    id_column = col
                    break

            if id_column:

                rows = metadata[
                    metadata[id_column]
                    .astype(str)
                    .str.strip()
                    == sample_id
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

                            path = resolve_project_path(
                                row[col]
                            )

                            if path:
                                return path

        except Exception:
            pass

    # --------------------------------------------------------
    # Common IAM locations
    # --------------------------------------------------------

    possible = [
        DATA_DIR / "raw" / "IAM" / "images" / f"{sample_id}.png",
        DATA_DIR / "raw" / "IAM" / "images" / f"{sample_id}.jpg",
        DATA_DIR / "raw" / "IAM" / "images" / f"{sample_id}.jpeg",
    ]

    for path in possible:

        if path.exists():
            return path

    iam_images = (
        DATA_DIR
        / "raw"
        / "IAM"
        / "images"
    )

    if iam_images.exists():

        for suffix in [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
        ]:

            matches = list(
                iam_images.rglob(
                    sample_id + suffix
                )
            )

            if matches:
                return matches[0]

    return None


# ============================================================
# WORD EXPLAINABILITY PATHS
# ============================================================

def get_sample_explainability_folder(sample_id):

    return (
        METADATA_DIR
        / f"{safe_filename(sample_id)}_word_explainability"
    )


def resolve_word_image(
    sample_id,
    word_id,
    word,
    rank,
    supplied_path=None,
    image_type="original",
):

    sample_id = normalize_id(sample_id)
    word_id = normalize_id(word_id)
    word = safe_text(word, "")

    folder = get_sample_explainability_folder(
        sample_id
    )

    if not folder.exists():
        return None

    # --------------------------------------------------------
    # Supplied path
    # --------------------------------------------------------

    if supplied_path:

        candidate = Path(
            str(supplied_path).strip()
        )

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
                folder_resolved = folder.resolve()

                if (
                    candidate.exists()
                    and folder_resolved in candidate.parents
                ):
                    return candidate

            except Exception:
                pass

    suffix = f"_{image_type}.png"

    # --------------------------------------------------------
    # Word ID
    # --------------------------------------------------------

    if word_id:

        matches = list(
            folder.glob(
                f"*_{word_id}{suffix}"
            )
        )

        if matches:
            return matches[0]

        matches = list(
            folder.glob(
                f"*{word_id}*{suffix}"
            )
        )

        if matches:
            return matches[0]

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    try:

        rank_number = int(float(rank))

        prefix = f"{rank_number:02d}_"

        matches = list(
            folder.glob(
                f"{prefix}*{suffix}"
            )
        )

        if matches:
            return matches[0]

    except Exception:
        pass

    # --------------------------------------------------------
    # Word text
    # --------------------------------------------------------

    clean_word = safe_filename(word)

    if clean_word and clean_word != "unknown":

        matches = list(
            folder.glob(
                f"*_{clean_word}{suffix}"
            )
        )

        if matches:
            return matches[0]

        matches = list(
            folder.glob(
                f"*{clean_word}*{suffix}"
            )
        )

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

        print(
            f"WARNING: Could not read {path}: {exc}"
        )

        return pd.DataFrame()


def load_data():

    return (
        load_csv(MASTER_FILE),
        load_csv(ANOMALY_FILE),
        load_csv(WORD_ANOMALY_FILE),
        load_csv(WORD_SUMMARY_FILE),
        load_csv(METADATA_ALL_FILE),
        load_csv(WORDS_METADATA_FILE),
    )


# ============================================================
# SAMPLE ROW
# ============================================================

def get_row_from_file(
    df,
    sample_id,
    id_columns,
):

    if df.empty:
        return None

    for col in id_columns:

        if col not in df.columns:
            continue

        rows = df[
            df[col]
            .astype(str)
            .str.strip()
            == sample_id
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

    row = {}

    sources = [
        (
            metadata,
            [
                "sample_id",
                "form_id",
                "id",
            ],
        ),
        (
            anomaly,
            [
                "sample_id",
                "form_id",
                "id",
            ],
        ),
        (
            word_summary,
            [
                "sample_id",
                "form_id",
                "id",
            ],
        ),
        (
            master,
            [
                "sample_id",
                "form_id",
                "id",
            ],
        ),
    ]

    for df, id_columns in sources:

        source_row = get_row_from_file(
            df,
            sample_id,
            id_columns,
        )

        if source_row is not None:
            row.update(
                source_row.to_dict()
            )

    return pd.Series(row)


# ============================================================
# LOF FEATURE SENSITIVITY
# ============================================================

def load_lof_feature_sensitivity(sample_id):

    sensitivity_file = (
        METADATA_DIR
        / f"{safe_filename(sample_id)}"
        "_lof_feature_sensitivity.csv"
    )

    if not sensitivity_file.exists():

        return (
            pd.DataFrame(),
            sensitivity_file,
        )

    try:

        df = pd.read_csv(
            sensitivity_file
        )

        # ----------------------------------------------------
        # CURRENT CSV SCHEMA
        # ----------------------------------------------------

        required_columns = [
            "feature",
            "original_value",
            "reference_value",
            "original_lof_score",
            "lof_score_after_replacement",
            "lof_score_change",
            "relative_lof_sensitivity_pct",
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            print(
                "WARNING: LOF sensitivity file "
                f"is missing columns: {missing}"
            )

            return (
                pd.DataFrame(),
                sensitivity_file,
            )

        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        numeric_columns = [
            "original_value",
            "reference_value",
            "original_lof_score",
            "lof_score_after_replacement",
            "lof_score_change",
            "relative_lof_sensitivity_pct",
        ]

        for col in numeric_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        df = df.dropna(
            subset=[
                "lof_score_change",
                "relative_lof_sensitivity_pct",
            ]
        ).copy()

        if df.empty:
            return (
                df,
                sensitivity_file,
            )

        # ----------------------------------------------------
        # ABSOLUTE LOF CHANGE
        #
        # Use the CSV value if present.
        # Otherwise calculate it.
        # ----------------------------------------------------

        if "absolute_lof_change" in df.columns:

            df["absolute_lof_change"] = pd.to_numeric(
                df["absolute_lof_change"],
                errors="coerce",
            )

        else:

            df["absolute_lof_change"] = (
                df["lof_score_change"].abs()
            )

        # Fill any missing absolute values safely.

        df["absolute_lof_change"] = (
            df["absolute_lof_change"]
            .fillna(
                df["lof_score_change"].abs()
            )
        )

        # ----------------------------------------------------
        # SORT BY ACTUAL MEASURED LOF EFFECT
        # ----------------------------------------------------

        df = df.sort_values(
            "absolute_lof_change",
            ascending=False,
        ).reset_index(
            drop=True
        )

        df["report_rank"] = np.arange(
            1,
            len(df) + 1,
        )

        return (
            df,
            sensitivity_file,
        )

    except Exception as exc:

        print(
            f"WARNING: Could not load "
            f"{sensitivity_file}: {exc}"
        )

        return (
            pd.DataFrame(),
            sensitivity_file,
        )


# ============================================================
# LOF SENSITIVITY SECTION
# ============================================================

def create_lof_sensitivity_section(
    sample_id,
    sensitivity_df=None,
):

    if sensitivity_df is None:

        sensitivity_df, sensitivity_file = (
            load_lof_feature_sensitivity(
                sample_id
            )
        )

    else:

        sensitivity_file = (
            METADATA_DIR
            / f"{safe_filename(sample_id)}"
            "_lof_feature_sensitivity.csv"
        )

    if sensitivity_df.empty:

        return f"""
        <section id="lof-sensitivity">

            <div class="section-header">
                <div class="section-number">05</div>
                <div>
                    <div class="eyebrow">Feature Sensitivity</div>
                    <h2>LOF Feature Sensitivity Analysis</h2>
                </div>
            </div>

            <div class="warning">
                <div class="status-icon">!</div>
                <div>
                    <strong>
                        LOF feature sensitivity data not available.
                    </strong>

                    <p>
                        No valid sample-specific LOF sensitivity
                        records were available for
                        <strong>{html_escape(sample_id)}</strong>.
                    </p>

                    <p>
                        Expected file:
                        <code>{html_escape(sensitivity_file.name)}</code>
                    </p>
                </div>
            </div>

        </section>
        """

    top = sensitivity_df.iloc[0]

    top_feature = safe_text(
        top.get("feature"),
        "N/A",
    )

    top_label = feature_label(
        top_feature,
        FORM_LOF_FEATURES,
    )

    top_original = safe_float(
        top.get("original_value")
    )

    top_reference = safe_float(
        top.get("reference_value")
    )

    top_lof_original = safe_float(
        top.get("original_lof_score")
    )

    top_lof_modified = safe_float(
        top.get(
            "lof_score_after_replacement"
        )
    )

    top_lof_change = safe_float(
        top.get("lof_score_change")
    )

    top_sensitivity = safe_float(
        top.get(
            "relative_lof_sensitivity_pct"
        )
    )

    direction_text = (
        "decreased"
        if top_lof_change < 0
        else "increased"
        if top_lof_change > 0
        else "did not change"
    )

    direction_class = (
        "negative"
        if top_lof_change < 0
        else "positive"
        if top_lof_change > 0
        else "neutral"
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    table_rows = []

    for _, item in sensitivity_df.iterrows():

        feature = safe_text(
            item.get("feature"),
            "N/A",
        )

        label = feature_label(
            feature,
            FORM_LOF_FEATURES,
        )

        original = safe_float(
            item.get("original_value")
        )

        reference = safe_float(
            item.get("reference_value")
        )

        lof_original = safe_float(
            item.get("original_lof_score")
        )

        lof_modified = safe_float(
            item.get(
                "lof_score_after_replacement"
            )
        )

        lof_change = safe_float(
            item.get("lof_score_change")
        )

        sensitivity = safe_float(
            item.get(
                "relative_lof_sensitivity_pct"
            )
        )

        if lof_change < 0:
            direction = "Decreased"
            direction_class = "negative"

        elif lof_change > 0:
            direction = "Increased"
            direction_class = "positive"

        else:
            direction = "No change"
            direction_class = "neutral"

        table_rows.append(
            f"""
            <tr>

                <td class="rank-cell">
                    <span class="rank-pill">
                        {int(item["report_rank"])}
                    </span>
                </td>

                <td>
                    <strong>{html_escape(label)}</strong>
                    <span class="feature-code">
                        {html_escape(feature)}
                    </span>
                </td>

                <td>
                    {format_number(original, 5)}
                </td>

                <td>
                    {format_number(reference, 5)}
                </td>

                <td>
                    {format_number(lof_original, 6)}
                </td>

                <td>
                    {format_number(lof_modified, 6)}
                </td>

                <td>
                    <strong>
                        {format_number(lof_change, 6)}
                    </strong>
                </td>

                <td>
                    <span class="direction {direction_class}">
                        {direction}
                    </span>
                </td>

                <td>
                    <strong>
                        {format_percent(sensitivity, 2)}
                    </strong>
                </td>

            </tr>
            """
        )

    # --------------------------------------------------------
    # SECOND FEATURE
    # --------------------------------------------------------

    second_text = ""

    if len(sensitivity_df) >= 2:

        second = sensitivity_df.iloc[1]

        second_feature = safe_text(
            second.get("feature"),
            "N/A",
        )

        second_label = feature_label(
            second_feature,
            FORM_LOF_FEATURES,
        )

        second_change = safe_float(
            second.get(
                "lof_score_change"
            )
        )

        second_sensitivity = safe_float(
            second.get(
                "relative_lof_sensitivity_pct"
            )
        )

        second_text = (
            f"The second-largest measured LOF sensitivity was "
            f"<strong>{html_escape(second_label)}</strong>, "
            f"with an LOF-score change of "
            f"<strong>{format_number(second_change, 6)}</strong> "
            f"and a relative sensitivity of "
            f"<strong>{format_percent(second_sensitivity, 2)}</strong>."
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    if top_lof_change < 0:

        primary_interpretation = (
            f"Replacing <strong>{html_escape(top_label)}</strong> "
            f"with its training-reference value "
            f"<strong>{direction_text}</strong> the LOF anomaly "
            f"score from "
            f"<strong>{format_number(top_lof_original, 6)}</strong> "
            f"to "
            f"<strong>{format_number(top_lof_modified, 6)}</strong>. "
            f"The measured reduction was "
            f"<strong>{format_number(abs(top_lof_change), 6)}</strong>."
        )

    elif top_lof_change > 0:

        primary_interpretation = (
            f"Replacing <strong>{html_escape(top_label)}</strong> "
            f"with its training-reference value "
            f"<strong>{direction_text}</strong> the LOF anomaly "
            f"score from "
            f"<strong>{format_number(top_lof_original, 6)}</strong> "
            f"to "
            f"<strong>{format_number(top_lof_modified, 6)}</strong>. "
            f"The measured increase was "
            f"<strong>{format_number(abs(top_lof_change), 6)}</strong>."
        )

    else:

        primary_interpretation = (
            f"Replacing <strong>{html_escape(top_label)}</strong> "
            "with its training-reference value produced no measurable "
            "change in the LOF anomaly score."
        )

    return f"""
    <section id="lof-sensitivity">

        <div class="section-header">
            <div class="section-number">05</div>
            <div>
                <div class="eyebrow">Feature Sensitivity</div>
                <h2>LOF Feature Sensitivity Analysis</h2>
            </div>
        </div>

        <p class="lead">
            This controlled experiment examines how sensitive the
            form-level LOF anomaly score is to each of the nine
            selected OpenCV handwriting features.
        </p>

        <p>
            One feature at a time is replaced with its training-set
            reference value while the remaining sample features are
            kept unchanged. The modified feature vector is passed
            through the same learned representation pipeline and
            the LOF score is recalculated.
        </p>

        <div class="sensitivity-hero">

            <div class="sensitivity-label">
                MOST LOF-SENSITIVE FEATURE
            </div>

            <div class="sensitivity-feature">
                {html_escape(top_label)}
            </div>

            <div class="sensitivity-description">
                Replacing this feature with its training reference
                value {direction_text} the measured LOF score.
            </div>

            <div class="sensitivity-metrics">

                <div class="sensitivity-metric">
                    <span>Original Value</span>
                    <strong>
                        {format_number(top_original, 5)}
                    </strong>
                </div>

                <div class="sensitivity-metric">
                    <span>Training Reference</span>
                    <strong>
                        {format_number(top_reference, 5)}
                    </strong>
                </div>

                <div class="sensitivity-metric">
                    <span>Original LOF</span>
                    <strong>
                        {format_number(top_lof_original, 6)}
                    </strong>
                </div>

                <div class="sensitivity-metric">
                    <span>Modified LOF</span>
                    <strong>
                        {format_number(top_lof_modified, 6)}
                    </strong>
                </div>

                <div class="sensitivity-metric">
                    <span>LOF Score Change</span>
                    <strong>
                        {format_number(top_lof_change, 6)}
                    </strong>
                </div>

                <div class="sensitivity-metric emphasis">
                    <span>Relative Sensitivity</span>
                    <strong>
                        {format_percent(top_sensitivity, 2)}
                    </strong>
                </div>

            </div>

        </div>

        <h3>Feature Sensitivity Ranking</h3>

        <div class="table-wrapper">

            <table class="sensitivity-table">

                <thead>

                    <tr>
                        <th>Rank</th>
                        <th>Form Feature</th>
                        <th>Original</th>
                        <th>Reference</th>
                        <th>Original LOF</th>
                        <th>Modified LOF</th>
                        <th>LOF Change</th>
                        <th>Direction</th>
                        <th>Sensitivity</th>
                    </tr>

                </thead>

                <tbody>
                    {''.join(table_rows)}
                </tbody>

            </table>

        </div>

        <h3>Sample-Specific Interpretation</h3>

        <div class="interpretation-box">
            <p>
                {primary_interpretation}
            </p>

            <p>
                Among the nine one-feature-at-a-time perturbations,
                <strong>{html_escape(top_label)}</strong> produced the
                largest absolute change in the form-level LOF anomaly
                score, with a relative sensitivity of
                <strong>{format_percent(top_sensitivity, 2)}</strong>.
            </p>

            <p>
                {second_text}
            </p>
        </div>

        <div class="note">

            <strong>How to interpret this analysis</strong>

            <p>
                The sensitivity percentage represents this feature's
                share of the total absolute LOF-score change measured
                across the nine separate perturbation experiments.
            </p>

            <p>
                It does <strong>not</strong> represent a probability,
                confidence, severity percentage, diagnostic measure,
                or proof that the feature caused the anomaly.
            </p>

        </div>

        <div class="note">

            <strong>Important limitation</strong>

            <p>
                This is a one-feature-at-a-time sensitivity analysis.
                It measures the effect of individually replacing
                features with their training-reference values. It
                does not capture interactions between multiple
                OpenCV features.
            </p>

        </div>

        <p class="source-note">
            Sensitivity source:
            <code>{html_escape(sensitivity_file.name)}</code>
        </p>

    </section>
    """


# ============================================================
# WORD EXPLAINABILITY
# ============================================================

def load_word_explainability(sample_id):

    folder = get_sample_explainability_folder(
        sample_id
    )

    report_file = (
        folder
        / "word_explainability_report.csv"
    )

    if not report_file.exists():
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            report_file
        )

        if "sample_id" in df.columns:

            df = df[
                df["sample_id"]
                .astype(str)
                .str.strip()
                == sample_id
            ]

        return df

    except Exception as exc:

        print(
            f"WARNING: Could not read "
            f"{report_file}: {exc}"
        )

        return pd.DataFrame()


def get_word_id_column(df):

    for col in [
        "word_id",
        "id",
    ]:

        if col in df.columns:
            return col

    return None


def filter_word_anomaly_for_sample(
    word_df,
    sample_id,
):

    if word_df.empty:
        return pd.DataFrame()

    for col in [
        "form_id",
        "sample_id",
    ]:

        if col in word_df.columns:

            result = word_df[
                word_df[col]
                .astype(str)
                .str.strip()
                == sample_id
            ].copy()

            if not result.empty:
                return result

    return pd.DataFrame(
        columns=word_df.columns
    )


# ============================================================
# WORD TEXT LOOKUP
# ============================================================

def lookup_word_text(
    word_id,
    words_metadata,
):

    if (
        words_metadata.empty
        or not word_id
    ):
        return None

    id_column = get_word_id_column(
        words_metadata
    )

    if not id_column:
        return None

    rows = words_metadata[
        words_metadata[id_column]
        .astype(str)
        .str.strip()
        == str(word_id).strip()
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

            value = safe_text(
                row[col],
                "",
            )

            if value:
                return value

    return None


# ============================================================
# DATASET PERCENTILES
# ============================================================

def percentile_rank(
    value,
    values,
):

    value = safe_float(value)

    if not np.isfinite(value):
        return np.nan

    clean = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    return float(
        (
            clean <= value
        ).mean()
        * 100
    )


def get_dataset_percentiles(
    sample_id,
    master,
    word_summary,
):

    result = {
        "lof_percentile": np.nan,
        "word_percentile": np.nan,
    }

    # --------------------------------------------------------
    # LOF
    # --------------------------------------------------------

    if not master.empty:

        sample_rows = pd.DataFrame()

        if "sample_id" in master.columns:

            sample_rows = master[
                master["sample_id"]
                .astype(str)
                .str.strip()
                == sample_id
            ]

        elif "form_id" in master.columns:

            sample_rows = master[
                master["form_id"]
                .astype(str)
                .str.strip()
                == sample_id
            ]

        if not sample_rows.empty:

            row = sample_rows.iloc[0]

            if "resnet_rank_percentile" in row and np.isfinite(safe_float(row.get("resnet_rank_percentile"))):

                result["lof_percentile"] = safe_float(
                    row.get("resnet_rank_percentile")
                )

            elif "lof_anomaly_score" in master.columns:

                result["lof_percentile"] = (
                    percentile_rank(
                        row.get(
                            "lof_anomaly_score"
                        ),
                        master[
                            "lof_anomaly_score"
                        ],
                    )
                )

            elif "lof_raw_anomaly" in master.columns:

                result["lof_percentile"] = (
                    percentile_rank(
                        row.get(
                            "lof_raw_anomaly"
                        ),
                        master[
                            "lof_raw_anomaly"
                        ],
                    )
                )

            elif "resnet_raw_anomaly" in master.columns:

                result["lof_percentile"] = (
                    percentile_rank(
                        row.get(
                            "resnet_raw_anomaly"
                        ),
                        master[
                            "resnet_raw_anomaly"
                        ],
                    )
                )

            elif "resnet_anomaly_score" in master.columns:

                result["lof_percentile"] = (
                    percentile_rank(
                        row.get(
                            "resnet_anomaly_score"
                        ),
                        master[
                            "resnet_anomaly_score"
                        ],
                    )
                )

    # --------------------------------------------------------
    # WORD
    # --------------------------------------------------------

    if not word_summary.empty:

        sample_rows = pd.DataFrame()

        if "sample_id" in word_summary.columns:

            sample_rows = word_summary[
                word_summary["sample_id"]
                .astype(str)
                .str.strip()
                == sample_id
            ]

        elif "form_id" in word_summary.columns:

            sample_rows = word_summary[
                word_summary["form_id"]
                .astype(str)
                .str.strip()
                == sample_id
            ]

        if not sample_rows.empty:

            row = sample_rows.iloc[0]

            for col in [
                "mean_word_anomaly",
                "mean_anomaly",
                "word_anomaly_mean",
            ]:

                if col in word_summary.columns:

                    result["word_percentile"] = (
                        percentile_rank(
                            row.get(col),
                            word_summary[col],
                        )
                    )

                    break

    return result


# ============================================================
# AGREEMENT
# ============================================================

def get_agreement(
    row,
    percentiles,
):

    for col in [
        "anomaly_agreement",
        "agreement_category",
        "agreement",
    ]:

        if col in row.index:

            value = safe_text(
                row.get(col),
                "",
            )

            if value:
                return value

    lp = percentiles[
        "lof_percentile"
    ]

    wp = percentiles[
        "word_percentile"
    ]

    if (
        not np.isfinite(lp)
        or not np.isfinite(wp)
    ):

        return "Not available"

    if lp >= 75 and wp >= 75:
        return "High in both"

    if lp >= 75 and wp < 75:
        return "High LOF only"

    if lp < 75 and wp >= 75:
        return "High word anomaly only"

    return "Low/moderate in both"


# ============================================================
# TOP WORDS
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
        subset=[
            "word_anomaly_score"
        ]
    )

    word_df = word_df.sort_values(
        "word_anomaly_score",
        ascending=False,
    ).head(number)

    word_id_col = get_word_id_column(
        word_df
    )

    if word_id_col:

        word_df["_join_word_id"] = (
            word_df[word_id_col]
            .astype(str)
            .str.strip()
        )

    else:

        word_df["_join_word_id"] = ""

    if not explanation_df.empty:

        explanation_df = explanation_df.copy()

        explanation_id_col = (
            get_word_id_column(
                explanation_df
            )
        )

        if explanation_id_col:

            explanation_df["_join_word_id"] = (
                explanation_df[
                    explanation_id_col
                ]
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
                col
                for col in keep_cols
                if col in explanation_df.columns
            ]

            explanation_small = (
                explanation_df[
                    keep_cols
                ]
                .drop_duplicates(
                    "_join_word_id"
                )
            )

            word_df = word_df.merge(
                explanation_small,
                on="_join_word_id",
                how="left",
                suffixes=(
                    "",
                    "_explanation",
                ),
            )

    words = []

    for _, row in word_df.iterrows():

        word = ""

        if "word" in row.index:

            word = safe_text(
                row["word"],
                "",
            )

        if (
            not word
            and "word_explanation" in row.index
        ):

            word = safe_text(
                row["word_explanation"],
                "",
            )

        if not word:

            word_id = safe_text(
                row.get(
                    "_join_word_id"
                ),
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


def dominant_word_features(
    top_words,
):

    if top_words.empty:
        return []

    counts = {}

    for _, row in top_words.iterrows():

        feature = safe_text(
            row.get(
                "main_anomaly_feature"
            ),
            "",
        )

        if feature:

            counts[feature] = (
                counts.get(feature, 0)
                + 1
            )

    return sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )


# ============================================================
# FEATURE INTERPRETATION
# ============================================================

def interpret_sample_feature(
    feature,
    value,
):

    value = safe_float(value)

    if not np.isfinite(value):
        return "Measurement unavailable."

    explanations = {

        "skew":
            "Estimated orientation or skew of the writing layout.",

        "baseline_deviation":
            "Variation in the estimated writing baseline.",

        "word_spacing_cv":
            "Relative variability in spacing between detected words.",

        "character_height_cv":
            "Relative variability in character height.",

        "average_word_height":
            "Average vertical size of detected word regions.",

        "average_word_width":
            "Average horizontal size of detected word regions.",

        "stroke_density":
            "Proportion of the writing region occupied by detected ink.",

        "slant_angle":
            "Estimated directional inclination of handwriting strokes.",

        "writing_area":
            "Proportion of the normalized form occupied by writing.",

        "connected_component_count":
            "Number of connected ink components detected in the sample.",

        "num_lines":
            "Number of handwriting lines detected in the form.",

        "num_words":
            "Number of word regions detected during form segmentation.",

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
# SAMPLE FINDINGS
# ============================================================

def build_sample_findings(
    sample_id,
    row,
    top_words,
    percentiles,
    agreement,
    lof_sensitivity_df,
):

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

    highest_z = np.nan
    highest_z_feature = None

    if not top_words.empty:

        for _, word in top_words.iterrows():

            z = safe_float(
                word.get(
                    "main_feature_z"
                )
            )

            if np.isfinite(z):

                if (
                    not np.isfinite(highest_z)
                    or abs(z) > abs(highest_z)
                ):

                    highest_z = z

                    highest_z_feature = safe_text(
                        word.get(
                            "main_anomaly_feature"
                        ),
                        "",
                    )

    # --------------------------------------------------------
    # LOF FEATURE
    # --------------------------------------------------------

    lof_feature = None
    lof_feature_label = None
    lof_feature_change = np.nan
    lof_feature_sensitivity = np.nan

    if (
        lof_sensitivity_df is not None
        and not lof_sensitivity_df.empty
    ):

        top_lof = lof_sensitivity_df.iloc[0]

        lof_feature = safe_text(
            top_lof.get("feature"),
            "",
        )

        lof_feature_label = feature_label(
            lof_feature,
            FORM_LOF_FEATURES,
        )

        # IMPORTANT:
        # CURRENT CSV COLUMN NAMES

        lof_feature_change = safe_float(
            top_lof.get(
                "lof_score_change"
            )
        )

        lof_feature_sensitivity = safe_float(
            top_lof.get(
                "relative_lof_sensitivity_pct"
            )
        )

    # --------------------------------------------------------
    # AGREEMENT
    # --------------------------------------------------------

    if agreement == "High in both":

        headline = "GLOBAL + LOCAL ANOMALY EVIDENCE"

        finding = (
            f"Sample {sample_id} shows elevated anomaly evidence "
            "at both the whole-form representation level and the "
            "individual-word level."
        )

        interpretation = (
            "The LOF-based form-level analysis identifies the sample "
            "as relatively isolated in the learned 256-dimensional "
            "embedding space, while the word-level measurements "
            "also identify elevated deviations."
        )

    elif agreement == "High LOF only":

        headline = "FORM-LEVEL ANOMALY DOMINATES"

        finding = (
            f"Sample {sample_id} shows stronger anomaly evidence "
            "at the whole-form representation level than at the "
            "aggregate word level."
        )

        interpretation = (
            "The sample is relatively isolated from its local "
            "neighbourhood in the learned 256-dimensional embedding "
            "space, while aggregate word-level measurements show "
            "comparatively weaker deviation."
        )

    elif agreement == "High word anomaly only":

        headline = "WORD-LEVEL ANOMALY DOMINATES"

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

        headline = "LOWER ANOMALY AT BOTH LEVELS"

        finding = (
            f"Sample {sample_id} does not show strong anomaly evidence "
            "at either the whole-form or aggregate word level."
        )

        interpretation = (
            "The sample is comparatively close to the reference "
            "distribution used by the research pipeline."
        )

    # --------------------------------------------------------
    # WORD PATTERN
    # --------------------------------------------------------

    if dominant_feature:

        feature_label_text = feature_label(
            dominant_feature,
            WORD_FEATURES,
        )

        word_pattern = (
            f"The dominant word-level anomaly feature among the "
            f"top-ranked words is <strong>"
            f"{html_escape(feature_label_text)}</strong>, "
            f"appearing in {dominant_count} of the top "
            f"{len(top_words)} anomalous words."
        )

    else:

        word_pattern = (
            "A dominant word-level anomaly feature could not "
            "be determined from the available records."
        )

    # --------------------------------------------------------
    # Z-SCORE
    # --------------------------------------------------------

    if np.isfinite(highest_z):

        z_label = feature_label(
            highest_z_feature,
            WORD_FEATURES,
        )

        z_pattern = (
            f"The strongest absolute robust Z-score among the "
            f"top-ranked words is <strong>"
            f"{format_number(highest_z, 2)}</strong> "
            f"for {html_escape(z_label)}."
        )

    else:

        z_pattern = (
            "A robust Z-score was not available for the selected "
            "word-level records."
        )

    # --------------------------------------------------------
    # PERCENTILES
    # --------------------------------------------------------

    lp = percentiles[
        "lof_percentile"
    ]

    wp = percentiles[
        "word_percentile"
    ]

    percentile_text = []

    if np.isfinite(lp):

        percentile_text.append(
            f"the form-level LOF score is at approximately "
            f"the {format_percent(lp, 1)} dataset percentile"
        )

    if np.isfinite(wp):

        percentile_text.append(
            f"the mean word anomaly is at approximately "
            f"the {format_percent(wp, 1)} dataset percentile"
        )

    relative_position = (
        " and ".join(percentile_text) + "."
        if percentile_text
        else
        "dataset percentile information is not available."
    )

    # --------------------------------------------------------
    # LOF FEATURE TEXT
    # --------------------------------------------------------

    if lof_feature:

        if lof_feature_change < 0:
            lof_direction = "reduced"

        elif lof_feature_change > 0:
            lof_direction = "increased"

        else:
            lof_direction = "did not change"

        lof_feature_pattern = (
            f"The most LOF-sensitive form-level feature was "
            f"<strong>{html_escape(lof_feature_label)}</strong>. "
            f"Replacing it with its training-reference value "
            f"{lof_direction} the LOF score by "
            f"<strong>{format_number(abs(lof_feature_change), 6)}</strong>, "
            f"representing "
            f"{format_percent(lof_feature_sensitivity, 2)} "
            "of the total absolute LOF change measured across the "
            "nine one-feature-at-a-time perturbations."
        )

    else:

        lof_feature_pattern = (
            "Sample-specific LOF feature sensitivity information "
            "was not available."
        )

    # --------------------------------------------------------
    # WHAT THE SAMPLE TELLS US
    # --------------------------------------------------------

    common_text = (
        f"{lof_feature_pattern} "
        f"{word_pattern} "
        f"{z_pattern}"
    )

    if agreement == "High in both":

        what_it_tells = (
            f"The evidence for {sample_id} is distributed across "
            "both whole-form and local handwriting characteristics. "
            "The form-level LOF analysis identifies the sample as "
            "relatively unusual in the learned embedding space, "
            "while several individual words also show elevated "
            "deviations. "
            f"{common_text} "
            "Taken together, this represents agreement between "
            "the two anomaly-analysis views."
        )

    elif agreement == "High LOF only":

        what_it_tells = (
            f"The main signal for {sample_id} comes from the "
            "whole-form embedding and its local-density relationship "
            "rather than extreme aggregate word anomalies. "
            f"{common_text} "
            "The word-level evidence is comparatively weaker, "
            "demonstrating the value of separating whole-form "
            "representation from local word-level analysis."
        )

    elif agreement == "High word anomaly only":

        what_it_tells = (
            f"The main signal for {sample_id} is localized to "
            "individual word regions rather than strongly expressed "
            "across the complete handwriting form. "
            f"{common_text} "
            "This indicates that local analysis contributes "
            "information that may be less visible in the whole-form "
            "representation."
        )

    else:

        what_it_tells = (
            f"The available evidence for {sample_id} is relatively "
            "moderate at both analysis levels. "
            f"{common_text} "
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
        "lof_feature_pattern": lof_feature_pattern,
        "relative_position": relative_position,
        "what_it_tells": what_it_tells,
        "dominant_feature": dominant_feature,
        "dominant_count": dominant_count,
        "highest_z": highest_z,
        "highest_z_feature": highest_z_feature,
        "lof_feature": lof_feature,
        "lof_feature_label": lof_feature_label,
        "lof_feature_change": lof_feature_change,
        "lof_feature_sensitivity": lof_feature_sensitivity,
    }


# ============================================================
# HTML COMPONENTS
# ============================================================

def image_html(
    path,
    alt,
    class_name="report-image",
):

    encoded = image_to_base64(path)

    if not encoded:

        return f"""
        <div class="missing-image">

            <div class="missing-icon">IMG</div>

            <strong>
                {html_escape(alt)}
            </strong>

            <span>
                Visual not available for this exact sample/word.
            </span>

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
    <section id="finding">

        <div class="section-header">

            <div class="section-number">01</div>

            <div>
                <div class="eyebrow">Sample Finding</div>
                <h2>Sample-Specific Finding</h2>
            </div>

        </div>

        <div class="finding-hero">

            <div class="finding-badge">
                {html_escape(findings["headline"])}
            </div>

            <h3 class="sample-title">
                {html_escape(sample_id)}
            </h3>

            <p class="finding-main">
                {findings["finding"]}
            </p>

            <p>
                {findings["interpretation"]}
            </p>

            <div class="evidence-grid">

                <div class="evidence-card">
                    <span>Agreement</span>
                    <strong>
                        {html_escape(agreement)}
                    </strong>
                </div>

                <div class="evidence-card">
                    <span>Most LOF-Sensitive Feature</span>
                    <strong>
                        {html_escape(
                            findings["lof_feature_label"]
                            or "N/A"
                        )}
                    </strong>
                </div>

                <div class="evidence-card">
                    <span>Highest Word Z-score</span>
                    <strong>
                        {format_number(
                            findings["highest_z"],
                            2,
                        )}
                    </strong>
                </div>

            </div>

            <h3>Most LOF-Sensitive Feature</h3>

            <p>
                {findings["lof_feature_pattern"]}
            </p>

            <h3>What this sample tells us</h3>

            <p>
                {findings["what_it_tells"]}
            </p>

            <div class="note">

                <strong>Research interpretation only</strong>

                <p>
                    These findings describe statistical and
                    representational deviations from the reference
                    distributions. They do not establish dysgraphia
                    or any other clinical condition.
                </p>

            </div>

        </div>

    </section>
    """


def create_summary_cards(
    lof_score,
    mean_word,
    max_word,
    top5,
    combined,
    agreement,
):

    return f"""
    <div class="summary-grid">

        <div class="metric-card primary">
            <div class="metric-icon">L</div>
            <span>LOF Form-Level Anomaly</span>
            <strong>{format_number(lof_score, 2)}</strong>
            <small>Higher = more locally isolated</small>
        </div>

        <div class="metric-card">
            <div class="metric-icon">W</div>
            <span>Mean Word Anomaly</span>
            <strong>{format_number(mean_word, 3)}</strong>
            <small>Aggregate local evidence</small>
        </div>

        <div class="metric-card">
            <div class="metric-icon">M</div>
            <span>Maximum Word Anomaly</span>
            <strong>{format_number(max_word, 3)}</strong>
            <small>Highest individual word score</small>
        </div>

        <div class="metric-card">
            <div class="metric-icon">T</div>
            <span>Top-5 Mean Word Anomaly</span>
            <strong>{format_number(top5, 3)}</strong>
            <small>Average of strongest local signals</small>
        </div>

        <div class="metric-card">
            <div class="metric-icon">G</div>
            <span>Global–Local Evidence Score</span>
            <strong>{format_number(combined, 2)}</strong>
            <small>50/50 percentile heuristic</small>
        </div>

        <div class="metric-card agreement-card">
            <div class="metric-icon">A</div>
            <span>Agreement</span>
            <strong>{html_escape(agreement)}</strong>
            <small>Dataset-relative comparison</small>
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

                <td>
                    <strong>{html_escape(label)}</strong>
                    <span class="feature-code">
                        {html_escape(key)}
                    </span>
                </td>

                <td class="value-cell">
                    {format_number(value, 4)}
                </td>

                <td>
                    {html_escape(
                        interpret_sample_feature(
                            key,
                            value,
                        )
                    )}
                </td>

            </tr>
            """
        )

    return f"""
    <div class="table-wrapper">

        <table>

            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>What It Measures</th>
                </tr>
            </thead>

            <tbody>
                {''.join(rows)}
            </tbody>

        </table>

    </div>
    """


def create_word_feature_table(row):

    rows = []

    for feature, label in WORD_FEATURES:

        value = row.get(feature)

        z_column = f"{feature}_z"

        z = row.get(z_column)

        if not np.isfinite(
            safe_float(z)
        ):

            main_feature = safe_text(
                row.get(
                    "main_anomaly_feature"
                ),
                "",
            )

            if main_feature == feature:
                z = row.get(
                    "main_feature_z"
                )

        rows.append(
            f"""
            <tr>

                <td>{html_escape(label)}</td>

                <td>
                    {format_number(value, 4)}
                </td>

                <td>
                    {format_number(z, 3)}
                </td>

            </tr>
            """
        )

    return f"""
    <div class="table-wrapper">

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

    </div>
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

        feature_label_text = feature_label(
            main_feature,
            WORD_FEATURES,
        )

        explanation = (
            f"The word has a word anomaly score of "
            f"{format_number(score, 2)}. The primary feature "
            f"identified for this word is {feature_label_text}, "
            f"with a robust Z-score of "
            f"{format_number(z, 2)}. "
            "The Grad-CAM highlights regions that contribute "
            "strongly to the ResNet50 handwriting representation. "
            "This visualization is model evidence of unusual "
            "representation patterns and is not a clinical diagnosis."
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

    position_text = ""
    try:
        parts = word_id.split("-")
        if len(parts) >= 4:
            line_idx = int(parts[-2])
            word_idx = int(parts[-1])
            position_text = f"Line {line_idx + 1}, Word {word_idx + 1}"
    except Exception:
        pass

    return f"""
    <div class="word-card">

        <div class="word-card-header">

            <div>
                <div class="word-rank">#{rank}</div>
                <h3>
                    {html_escape(word)}
                    {f'<span class="instance-pill">{html_escape(position_text)}</span>' if position_text else ''}
                </h3>
            </div>

            <div class="word-score">
                <span>Anomaly Score</span>
                <strong>{format_number(score, 3)}</strong>
            </div>

        </div>

        <div class="word-meta">

            <span>
                <strong>Word ID</strong>
                {html_escape(word_id)}
            </span>

            <span>
                <strong>Form Location</strong>
                {html_escape(position_text if position_text else 'N/A')}
            </span>

            <span>
                <strong>Main Feature</strong>
                {html_escape(
                    feature_label(
                        main_feature,
                        WORD_FEATURES,
                    )
                )}
            </span>

            <span>
                <strong>Z-score</strong>
                {format_number(z, 3)}
            </span>

        </div>

        <div class="word-images">

            <div class="image-panel">
                <h4>Original Word</h4>
                {image_html(
                    original_path,
                    f"Original word {word}",
                )}
            </div>

            <div class="image-panel">
                <h4>ResNet50 Grad-CAM</h4>
                {image_html(
                    gradcam_path,
                    f"Grad-CAM for {word}",
                )}
            </div>

            <div class="image-panel">
                <h4>Original + Grad-CAM</h4>
                {image_html(
                    comparison_path,
                    f"Comparison for {word}",
                )}
            </div>

        </div>

        <details class="feature-details">

            <summary>
                Word-Level Feature Measurements
            </summary>

            {create_word_feature_table(row)}

        </details>

        <div class="activation-box">

            <strong>High-activation region:</strong>

            {format_percent(
                activation,
                2,
            )}

        </div>

        <div class="word-interpretation">

            <strong>Interpretation</strong>

            <p>
                {html_escape(explanation)}
            </p>

        </div>

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
    lof_sensitivity_df=None,
):

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    lof_score = safe_float(
        row.get("lof_anomaly_score")
    )

    if not np.isfinite(lof_score):

        lof_score = safe_float(
            row.get("lof_raw_anomaly")
        )

    if not np.isfinite(lof_score):

        lof_score = safe_float(
            row.get("resnet_raw_anomaly")
        )

    if not np.isfinite(lof_score):

        lof_score = safe_float(
            row.get("anomaly_score")
        )

    if not np.isfinite(lof_score):

        lof_score = safe_float(
            row.get("lof_score_samples")
        )

    if not np.isfinite(lof_score):

        lof_score = safe_float(
            row.get("resnet_anomaly_score")
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

        master_lof_series = []
        for col_cand in [
            "lof_anomaly_score",
            "lof_raw_anomaly",
            "resnet_raw_anomaly",
            "anomaly_score",
            "resnet_anomaly_score",
        ]:
            if col_cand in master.columns:
                master_lof_series = master[col_cand]
                break

        lp_temp = percentile_rank(
            lof_score,
            master_lof_series,
        )

        wp_temp = percentile_rank(
            mean_word,
            word_summary.get(
                "mean_word_anomaly",
                [],
            ),
        )

        if (
            np.isfinite(lp_temp)
            and np.isfinite(wp_temp)
        ):

            combined = (
                0.5 * lp_temp
                + 0.5 * wp_temp
            )

    # --------------------------------------------------------
    # PERCENTILES
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
    # LOF SENSITIVITY
    # --------------------------------------------------------

    if lof_sensitivity_df is None:

        lof_sensitivity_df, _ = (
            load_lof_feature_sensitivity(
                sample_id
            )
        )

    # --------------------------------------------------------
    # WORD EXPLAINABILITY
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
    # FINDINGS
    # --------------------------------------------------------

    findings = build_sample_findings(
        sample_id,
        row,
        top_words,
        percentiles,
        agreement,
        lof_sensitivity_df,
    )

    # --------------------------------------------------------
    # ORIGINAL SAMPLE
    # --------------------------------------------------------

    original_image = find_exact_sample_image(
        sample_id
    )

    original_section = image_html(
        original_image,
        f"Original handwriting sample {sample_id}",
        "sample-image",
    )

    # --------------------------------------------------------
    # WORD COUNTS
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
    # WORD FEATURE FREQUENCY
    # --------------------------------------------------------

    feature_counts = dominant_word_features(
        top_words
    )

    feature_rows = []

    for feature, count in feature_counts:

        label = feature_label(
            feature,
            WORD_FEATURES,
        )

        feature_rows.append(
            f"""
            <tr>
                <td>
                    <strong>
                        {html_escape(label)}
                    </strong>
                </td>
                <td>
                    <span class="count-pill">
                        {count}
                    </span>
                </td>
            </tr>
            """
        )

    # --------------------------------------------------------
    # WORD SECTIONS
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
    # CHART
    # --------------------------------------------------------

    chart_section = image_html(
        COMPARISON_CHART,
        "Dataset-level LOF versus word anomaly comparison",
    )

    # --------------------------------------------------------
    # EXPLAINABILITY STATUS
    # --------------------------------------------------------

    if explanation_df.empty:

        explainability_status = """
        <div class="warning">

            <div class="status-icon">!</div>

            <div>
                <strong>
                    Word explainability data unavailable
                </strong>

                <p>
                    Word-level anomaly results may still be shown,
                    but sample-specific Grad-CAM records were not
                    found for this sample.
                </p>
            </div>

        </div>
        """

    else:

        explainability_status = f"""
        <div class="success">

            <div class="status-icon">✓</div>

            <div>
                <strong>
                    Word explainability data available
                </strong>

                <p>
                    {explanation_count} explainability records
                    are associated with
                    <strong>{html_escape(sample_id)}</strong>.
                </p>
            </div>

        </div>
        """

    # --------------------------------------------------------
    # LOF STATUS
    # --------------------------------------------------------

    if lof_sensitivity_df.empty:

        lof_sensitivity_status = """
        <div class="warning">

            <div class="status-icon">!</div>

            <div>
                <strong>
                    LOF feature sensitivity unavailable
                </strong>

                <p>
                    The form-level LOF score is available, but
                    feature-level LOF sensitivity could not be
                    reported.
                </p>
            </div>

        </div>
        """

    else:

        top_feature = safe_text(
            lof_sensitivity_df.iloc[0].get(
                "feature"
            ),
            "N/A",
        )

        top_label = feature_label(
            top_feature,
            FORM_LOF_FEATURES,
        )

        lof_sensitivity_status = f"""
        <div class="success">

            <div class="status-icon">✓</div>

            <div>
                <strong>
                    LOF feature sensitivity data available
                </strong>

                <p>
                    Most LOF-sensitive feature:
                    <strong>
                        {html_escape(top_label)}
                    </strong>
                </p>
            </div>

        </div>
        """

    # --------------------------------------------------------
    # PERCENTILES
    # --------------------------------------------------------

    lp = percentiles[
        "lof_percentile"
    ]

    wp = percentiles[
        "word_percentile"
    ]

    # --------------------------------------------------------
    # HTML DOCUMENT
    # --------------------------------------------------------

    return f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Explainable Handwriting Anomaly Report —
{html_escape(sample_id)}
</title>

<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

:root {{
    --ink: #172033;
    --muted: #64748b;
    --soft: #f5f7fb;
    --line: #e2e8f0;
    --panel: #ffffff;
    --blue: #2563eb;
    --blue-soft: #eff6ff;
    --purple: #7c3aed;
    --purple-soft: #f5f3ff;
    --green: #16a34a;
    --green-soft: #f0fdf4;
    --orange: #ea580c;
    --orange-soft: #fff7ed;
    --red: #dc2626;
    --shadow:
        0 10px 30px rgba(15, 23, 42, 0.07);
}}

* {{
    box-sizing: border-box;
}}

html {{
    scroll-behavior: smooth;
}}

body {{
    margin: 0;
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    background:
        linear-gradient(
            180deg,
            #eef3f9 0,
            #f7f9fc 300px,
            #f7f9fc 100%
        );
    color: var(--ink);
    line-height: 1.7;
}}

.container {{
    max-width: 1280px;
    margin: auto;
    padding: 32px 24px 60px;
}}

p {{
    color: #475569;
}}

strong {{
    color: #172033;
}}

h1,
h2,
h3,
h4 {{
    color: var(--ink);
    line-height: 1.25;
}}

h2 {{
    margin: 0;
    font-size: 25px;
}}

h3 {{
    margin-top: 30px;
    font-size: 19px;
}}

code {{
    font-family:
        "Cascadia Code",
        "Consolas",
        monospace;
    font-size: 0.88em;
    background: #eef2f7;
    border: 1px solid #e2e8f0;
    padding: 3px 7px;
    border-radius: 6px;
    color: #334155;
    word-break: break-word;
}}


/* ==========================================================
   HEADER
   ========================================================== */

.report-header {{
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 55%,
            #263b61 100%
        );
    color: white;
    border-radius: 22px;
    padding: 42px;
    box-shadow:
        0 20px 45px rgba(15, 23, 42, 0.20);
    margin-bottom: 18px;
}}

.report-header::after {{
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    right: -100px;
    top: -150px;
    background: rgba(255,255,255,0.05);
}}

.report-header h1 {{
    position: relative;
    z-index: 1;
    color: white;
    margin: 0 0 12px;
    font-size: clamp(28px, 4vw, 44px);
    letter-spacing: -1px;
}}

.header-subtitle {{
    position: relative;
    z-index: 1;
    color: #cbd5e1;
    font-size: 16px;
    max-width: 760px;
}}

.header-meta {{
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 25px;
}}

.header-chip {{
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.09);
    border: 1px solid rgba(255,255,255,0.14);
    color: #e2e8f0;
    font-size: 12px;
    font-weight: 700;
}}


/* ==========================================================
   NAVIGATION
   ========================================================== */

.report-nav {{
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    background: rgba(255,255,255,0.92);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 10px;
    margin-bottom: 25px;
    position: sticky;
    top: 10px;
    z-index: 10;
    backdrop-filter: blur(12px);
    box-shadow: 0 5px 20px rgba(15,23,42,0.06);
}}

.report-nav a {{
    text-decoration: none;
    color: #475569;
    padding: 7px 10px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
}}

.report-nav a:hover {{
    background: var(--blue-soft);
    color: var(--blue);
}}


/* ==========================================================
   SECTION
   ========================================================== */

section {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 25px;
    box-shadow: var(--shadow);
}}

.section-header {{
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 22px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--line);
}}

.section-number {{
    flex: 0 0 auto;
    width: 48px;
    height: 48px;
    border-radius: 13px;
    display: grid;
    place-items: center;
    background: var(--blue-soft);
    color: var(--blue);
    font-size: 14px;
    font-weight: 900;
}}

.eyebrow {{
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--blue);
    font-size: 10px;
    font-weight: 900;
    margin-bottom: 4px;
}}

.lead {{
    font-size: 17px;
    color: #334155;
}}


/* ==========================================================
   SUMMARY CARDS
   ========================================================== */

.summary-grid {{
    display: grid;
    grid-template-columns:
        repeat(3, minmax(0, 1fr));
    gap: 15px;
}}

.metric-card {{
    position: relative;
    min-height: 145px;
    padding: 21px;
    border: 1px solid var(--line);
    border-radius: 15px;
    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f8fafc
        );
}}

.metric-card.primary {{
    border-color: #bfdbfe;
    background:
        linear-gradient(
            145deg,
            #eff6ff,
            #ffffff
        );
}}

.metric-icon {{
    width: 30px;
    height: 30px;
    display: grid;
    place-items: center;
    border-radius: 9px;
    background: #e2e8f0;
    color: #475569;
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 13px;
}}

.metric-card.primary .metric-icon {{
    background: #dbeafe;
    color: var(--blue);
}}

.metric-card span {{
    display: block;
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
}}

.metric-card strong {{
    display: block;
    margin-top: 5px;
    font-size: 25px;
    letter-spacing: -0.5px;
}}

.metric-card small {{
    display: block;
    color: #94a3b8;
    font-size: 10px;
    margin-top: 5px;
}}

.agreement-card strong {{
    font-size: 18px;
}}


/* ==========================================================
   FINDING
   ========================================================== */

.finding-hero {{
    background:
        linear-gradient(
            145deg,
            #f8fafc,
            #eef4ff
        );
    border: 1px solid #dbeafe;
    border-left: 5px solid var(--blue);
    border-radius: 16px;
    padding: 27px;
}}

.finding-badge {{
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    background: #dbeafe;
    color: #1e40af;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .4px;
}}

.sample-title {{
    font-size: 28px;
    margin: 16px 0 8px;
}}

.finding-main {{
    font-size: 18px;
    color: #1e293b;
}}

.evidence-grid {{
    display: grid;
    grid-template-columns:
        repeat(3, minmax(0, 1fr));
    gap: 13px;
    margin: 24px 0;
}}

.evidence-card {{
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 17px;
}}

.evidence-card span {{
    display: block;
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
}}

.evidence-card strong {{
    display: block;
    margin-top: 6px;
    font-size: 17px;
}}


/* ==========================================================
   SENSITIVITY
   ========================================================== */

.sensitivity-hero {{
    background:
        linear-gradient(
            135deg,
            #faf5ff,
            #f5f3ff 55%,
            #ffffff
        );
    border: 1px solid #ddd6fe;
    border-left: 5px solid var(--purple);
    border-radius: 17px;
    padding: 27px;
    margin: 22px 0;
}}

.sensitivity-label {{
    color: #7c3aed;
    font-size: 10px;
    letter-spacing: 1.2px;
    font-weight: 900;
}}

.sensitivity-feature {{
    font-size: 29px;
    font-weight: 900;
    margin-top: 7px;
    color: #4c1d95;
}}

.sensitivity-description {{
    color: #64748b;
    margin: 7px 0 22px;
}}

.sensitivity-metrics {{
    display: grid;
    grid-template-columns:
        repeat(3, minmax(0, 1fr));
    gap: 11px;
}}

.sensitivity-metric {{
    background: rgba(255,255,255,0.82);
    border: 1px solid #e9d5ff;
    border-radius: 11px;
    padding: 14px;
}}

.sensitivity-metric span {{
    display: block;
    color: #64748b;
    font-size: 10px;
    font-weight: 700;
}}

.sensitivity-metric strong {{
    display: block;
    margin-top: 5px;
    font-size: 17px;
}}

.sensitivity-metric.emphasis {{
    background: #ede9fe;
    border-color: #c4b5fd;
}}

.sensitivity-metric.emphasis strong {{
    color: #6d28d9;
    font-size: 21px;
}}

.interpretation-box {{
    background: #f8fafc;
    border-radius: 13px;
    padding: 18px 20px;
    border: 1px solid var(--line);
}}


/* ==========================================================
   TABLES
   ========================================================== */

.table-wrapper {{
    width: 100%;
    overflow-x: auto;
    border: 1px solid var(--line);
    border-radius: 12px;
    margin: 15px 0 20px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    min-width: 700px;
}}

th {{
    background: #f1f5f9;
    color: #475569;
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .4px;
    white-space: nowrap;
}}

th,
td {{
    padding: 12px 13px;
    border-bottom: 1px solid var(--line);
    vertical-align: middle;
}}

tbody tr:last-child td {{
    border-bottom: 0;
}}

tbody tr:hover {{
    background: #f8fafc;
}}

.value-cell {{
    font-family:
        "Cascadia Code",
        "Consolas",
        monospace;
    font-weight: 700;
}}

.feature-code {{
    display: block;
    color: #94a3b8;
    font-family:
        "Cascadia Code",
        "Consolas",
        monospace;
    font-size: 10px;
    margin-top: 3px;
}}

.rank-cell {{
    width: 55px;
}}

.rank-pill {{
    display: inline-grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: #ede9fe;
    color: #6d28d9;
    font-size: 11px;
    font-weight: 900;
}}

.direction {{
    display: inline-block;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
}}

.direction.negative {{
    background: #ecfdf5;
    color: #047857;
}}

.direction.positive {{
    background: #fff7ed;
    color: #c2410c;
}}

.direction.neutral {{
    background: #f1f5f9;
    color: #64748b;
}}

.count-pill {{
    display: inline-block;
    min-width: 28px;
    padding: 3px 8px;
    text-align: center;
    border-radius: 999px;
    background: #dbeafe;
    color: #1d4ed8;
    font-weight: 800;
    font-size: 11px;
}}

.small-table {{
    font-size: 12px;
}}


/* ==========================================================
   WORD CARDS
   ========================================================== */

.word-card {{
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 23px;
    margin: 22px 0;
    background:
        linear-gradient(
            145deg,
            #ffffff,
            #fafcff
        );
    box-shadow:
        0 7px 20px rgba(15,23,42,0.045);
}}

.word-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;
}}

.word-card-header h3 {{
    margin: 2px 0 0;
    font-size: 23px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}}

.instance-pill {{
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    color: #2563eb;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    padding: 3px 9px;
    border-radius: 9999px;
    letter-spacing: 0.2px;
    vertical-align: middle;
}}

.word-rank {{
    color: var(--blue);
    font-size: 10px;
    font-weight: 900;
    letter-spacing: .7px;
}}

.word-score {{
    text-align: right;
    background: #eff6ff;
    border: 1px solid #dbeafe;
    border-radius: 11px;
    padding: 10px 14px;
}}

.word-score span {{
    display: block;
    color: #64748b;
    font-size: 9px;
    font-weight: 700;
}}

.word-score strong {{
    display: block;
    color: #1d4ed8;
    font-size: 19px;
}}

.word-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin: 15px 0 20px;
}}

.word-meta span {{
    padding: 6px 9px;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid var(--line);
    color: #64748b;
    font-size: 10px;
}}

.word-meta strong {{
    color: #334155;
    margin-right: 4px;
}}

.word-images {{
    display: grid;
    grid-template-columns:
        repeat(3, minmax(0, 1fr));
    gap: 14px;
}}

.image-panel {{
    min-width: 0;
}}

.image-panel h4 {{
    font-size: 12px;
    margin: 0 0 8px;
    color: #475569;
}}

.image-container {{
    background: #f1f5f9;
    border: 1px solid var(--line);
    border-radius: 11px;
    padding: 9px;
    text-align: center;
}}

.report-image {{
    max-width: 100%;
    max-height: 450px;
    object-fit: contain;
    border-radius: 7px;
}}

.sample-image {{
    display: block;
    max-width: 92%;
    max-height: 850px;
    margin: auto;
    border-radius: 8px;
}}

.missing-image {{
    min-height: 210px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 7px;
    color: #64748b;
    text-align: center;
    padding: 20px;
}}

.missing-icon {{
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: #e2e8f0;
    color: #64748b;
    font-size: 10px;
    font-weight: 900;
}}

.feature-details {{
    margin-top: 18px;
}}

.feature-details summary {{
    cursor: pointer;
    color: #475569;
    font-size: 12px;
    font-weight: 800;
}}

.activation-box {{
    margin-top: 15px;
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 11px 14px;
    font-size: 12px;
}}

.word-interpretation {{
    margin-top: 15px;
    background: #eff6ff;
    border-left: 4px solid var(--blue);
    border-radius: 9px;
    padding: 13px 15px;
}}

.word-interpretation p {{
    margin-bottom: 0;
    font-size: 13px;
}}


/* ==========================================================
   IMAGE / LEGEND
   ========================================================== */

.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin: 18px 0;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 10px;
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 8px;
    color: #64748b;
    font-size: 11px;
}}

.legend-box {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
}}


/* ==========================================================
   STATUS / NOTES
   ========================================================== */

.warning,
.success,
.note {{
    display: flex;
    gap: 12px;
    padding: 16px;
    border-radius: 11px;
    margin: 18px 0;
}}

.warning {{
    background: var(--orange-soft);
    border: 1px solid #fed7aa;
    color: #9a3412;
}}

.success {{
    background: var(--green-soft);
    border: 1px solid #bbf7d0;
    color: #166534;
}}

.note {{
    background: var(--blue-soft);
    border: 1px solid #bfdbfe;
    color: #1e40af;
}}

.status-icon {{
    flex: 0 0 auto;
    width: 28px;
    height: 28px;
    display: grid;
    place-items: center;
    border-radius: 8px;
    background: rgba(255,255,255,0.65);
    font-weight: 900;
}}

.warning p,
.success p,
.note p {{
    margin: 5px 0 0;
    color: inherit;
}}

.source-note {{
    color: #94a3b8;
    font-size: 11px;
}}


/* ==========================================================
   ARCHITECTURE
   ========================================================== */

.architecture {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}}

.arch-node {{
    width: min(600px, 100%);
    text-align: center;
    padding: 15px;
    border: 1px solid var(--line);
    border-radius: 11px;
    background: #f8fafc;
    font-weight: 800;
}}

.arch-node.primary {{
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #1e40af;
}}

.arch-node.purple {{
    background: #f5f3ff;
    border-color: #ddd6fe;
    color: #6d28d9;
}}

.arch-arrow {{
    color: #94a3b8;
    font-size: 20px;
    font-weight: 900;
}}

.arch-plus {{
    color: #94a3b8;
    font-weight: 900;
}}


/* ==========================================================
   FOOTER
   ========================================================== */

footer {{
    text-align: center;
    color: #64748b;
    padding: 30px 10px;
    font-size: 12px;
}}

.footer-title {{
    color: #334155;
    font-weight: 900;
    margin-bottom: 5px;
}}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media(max-width: 950px) {{

    .summary-grid,
    .evidence-grid {{
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }}

    .sensitivity-metrics {{
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }}

    .word-images {{
        grid-template-columns: 1fr;
    }}

}}

@media(max-width: 650px) {{

    .container {{
        padding: 15px 10px 40px;
    }}

    .report-header {{
        padding: 27px 20px;
        border-radius: 16px;
    }}

    section {{
        padding: 20px 16px;
        border-radius: 14px;
    }}

    .summary-grid,
    .evidence-grid,
    .sensitivity-metrics {{
        grid-template-columns: 1fr;
    }}

    .report-nav {{
        position: static;
    }}

    .word-card-header {{
        align-items: flex-start;
        flex-direction: column;
    }}

    .word-score {{
        text-align: left;
    }}

    .section-header {{
        align-items: flex-start;
    }}

}}

</style>

</head>


<body>

<div class="container">


<!-- ======================================================
     HEADER
     ====================================================== -->

<header class="report-header">

    <h1>
        Explainable Handwriting
        Anomaly Detection Report
    </h1>

    <div class="header-subtitle">

        Sample-specific analysis combining learned visual
        representation, handcrafted handwriting measurements,
        local-density anomaly detection, word-level analysis,
        feature sensitivity, and visual explanation.

    </div>

    <div class="header-meta">

        <span class="header-chip">
            Sample: {html_escape(sample_id)}
        </span>

        <span class="header-chip">
            ResNet50
        </span>

        <span class="header-chip">
            OpenCV Features
        </span>

        <span class="header-chip">
            256-D Embedding
        </span>

        <span class="header-chip">
            LOF
        </span>

        <span class="header-chip">
            Word Analysis
        </span>

        <span class="header-chip">
            Grad-CAM
        </span>

    </div>

</header>


<!-- ======================================================
     NAVIGATION
     ====================================================== -->

<nav class="report-nav">

    <a href="#finding">01 Finding</a>
    <a href="#summary">02 Summary</a>
    <a href="#original">03 Image</a>
    <a href="#opencv">04 OpenCV</a>
    <a href="#lof-sensitivity">05 Sensitivity</a>
    <a href="#architecture">06 Architecture</a>
    <a href="#words">07 Words</a>
    <a href="#top-words">08 Top Words</a>
    <a href="#gradcam">09 Grad-CAM</a>
    <a href="#global-local">10 Global–Local</a>
    <a href="#interpretation">11 Findings</a>
    <a href="#limitations">12 Limits</a>
    <a href="#conclusion">13 Conclusion</a>

</nav>


<!-- ======================================================
     RESEARCH SCOPE BANNER
     ====================================================== -->

<div class="note">

    <div class="status-icon">i</div>

    <div>

        <strong>
            Research Scope
        </strong>

        <p>
            This report presents explainable handwriting
            anomaly-analysis results. The system is a research
            framework and is not a standalone clinical diagnostic
            system. Scores describe deviations relative to the
            reference data used by the pipeline.
        </p>

    </div>

</div>


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

<section id="summary">

    <div class="section-header">

        <div class="section-number">02</div>

        <div>
            <div class="eyebrow">Overview</div>
            <h2>Executive Summary</h2>
        </div>

    </div>

    {create_summary_cards(
        lof_score,
        mean_word,
        max_word,
        top5,
        combined,
        agreement,
    )}

    <div class="note">

        <div class="status-icon">i</div>

        <div>

            <strong>
                Score interpretation
            </strong>

            <p>
                The LOF score describes how isolated the sample is
                relative to the local neighbourhood of the training
                reference embeddings.
            </p>

            <p>
                Word-level anomaly scores describe deviations of
                individual word measurements from their reference
                distribution.
            </p>

            <p>
                The Global–Local Evidence Score is a percentile-based
                research heuristic combining the two evidence streams.
            </p>

            <p>
                These values are not dysgraphia probabilities,
                clinical severity scores, confidence scores, or
                medical diagnoses.
            </p>

        </div>

    </div>

</section>


<!-- ======================================================
     SECTION 3
     ====================================================== -->

<section id="original">

    <div class="section-header">

        <div class="section-number">03</div>

        <div>
            <div class="eyebrow">Visual Input</div>
            <h2>Original Handwriting Sample</h2>
        </div>

    </div>

    <p>
        Exact handwriting image associated with sample
        <strong>{html_escape(sample_id)}</strong>.
    </p>

    {original_section}

</section>


<!-- ======================================================
     SECTION 4
     ====================================================== -->

<section id="opencv">

    <div class="section-header">

        <div class="section-number">04</div>

        <div>
            <div class="eyebrow">Handcrafted Features</div>
            <h2>Sample-Level OpenCV Measurements</h2>
        </div>

    </div>

    <p>
        These measurements are extracted from the complete
        handwriting form. They are separate from the word-level
        anomaly features and form part of the learned form-level
        representation.
    </p>

    {create_sample_features_table(row)}

</section>


<!-- ======================================================
     SECTION 5
     ====================================================== -->

{create_lof_sensitivity_section(
    sample_id,
    lof_sensitivity_df,
)}


<!-- ======================================================
     SECTION 6
     ====================================================== -->

<section id="architecture">

    <div class="section-header">

        <div class="section-number">06</div>

        <div>
            <div class="eyebrow">Model Pipeline</div>
            <h2>Feature and Model Architecture</h2>
        </div>

    </div>

    <div class="architecture">

        <div class="arch-node primary">
            Handwriting Image
        </div>

        <div class="arch-arrow">↓</div>

        <div class="arch-node">
            ResNet50 — 2048-D
        </div>

        <div class="arch-plus">+</div>

        <div class="arch-node">
            9 Selected Form-Level
            OpenCV Features
        </div>

        <div class="arch-arrow">↓</div>

        <div class="arch-node">
            OpenCV Encoder — 64-D
        </div>

        <div class="arch-arrow">↓</div>

        <div class="arch-node">
            Feature Fusion — 2112-D
        </div>

        <div class="arch-arrow">↓</div>

        <div class="arch-node purple">
            Feature Attention
        </div>

        <div class="arch-arrow">↓</div>

        <div class="arch-node purple">
            256-D Shared Representation
        </div>

        <div class="arch-arrow">↓</div>

        <div class="arch-node">
            Local Outlier Factor (LOF)
        </div>

        <div class="arch-arrow">↓</div>

        <div class="arch-node primary">
            Form-Level Anomaly Score
        </div>

    </div>

    <p>
        The ResNet50 branch provides a 2048-dimensional visual
        representation, while the nine OpenCV features are encoded
        into a 64-dimensional handcrafted-feature representation.
    </p>

    <p>
        These representations are fused into a 2112-dimensional
        vector, processed through feature attention, and transformed
        into a 256-dimensional embedding.
    </p>

    <p>
        LOF is then applied in this learned embedding space. It
        measures how isolated each held-out handwriting sample is
        relative to the local neighbourhood established by the
        training embeddings.
    </p>

    <div class="note">

        <div class="status-icon">i</div>

        <div>

            <strong>
                Current evaluation setup
            </strong>

            <p>
                The current form-level evaluation uses the
                1232-sample training split and evaluates the LOF
                scorer on 307 held-out validation samples.
            </p>

        </div>

    </div>

</section>


<!-- ======================================================
     SECTION 7
     ====================================================== -->

<section id="words">

    <div class="section-header">

        <div class="section-number">07</div>

        <div>
            <div class="eyebrow">Local Analysis</div>
            <h2>Word-Level Anomaly Analysis</h2>
        </div>

    </div>

    <p>
        The form-level pipeline detected approximately
        <strong>
            {format_number(form_word_count, 0)}
        </strong>
        words, while the word-anomaly pipeline analyzed
        <strong>
            {valid_word_count}
        </strong>
        valid handwriting words.
    </p>

    <p>
        These counts can differ because the word-level pipeline
        filters invalid, punctuation, and unsuitable word regions
        before anomaly scoring.
    </p>

    <p>
        The word-level pipeline evaluates individual handwriting
        regions using robust feature-based anomaly scoring. A
        reference distribution is constructed using feature medians
        and median absolute deviations (MAD), and robust Z-scores
        quantify feature-level deviation.
    </p>

    {explainability_status}

    {lof_sensitivity_status}

    <h3>
        Dominant Word-Level Pattern
    </h3>

    <p>
        {findings["word_pattern"]}
    </p>

    <p>
        {findings["z_pattern"]}
    </p>

    <div class="table-wrapper">

        <table>

            <thead>

                <tr>
                    <th>Recurring Feature</th>
                    <th>Occurrences in Top 10</th>
                </tr>

            </thead>

            <tbody>

                {''.join(feature_rows)
                if feature_rows
                else
                '<tr><td colspan="2">No dominant feature available.</td></tr>'}

            </tbody>

        </table>

    </div>

</section>


<!-- ======================================================
     SECTION 8
     ====================================================== -->

<section id="top-words">

    <div class="section-header">

        <div class="section-number">08</div>

        <div>
            <div class="eyebrow">Local Evidence</div>
            <h2>Top 10 Anomalous Words</h2>
        </div>

    </div>

    <p>
        The following words are ranked by their word-level anomaly
        score within the selected sample. Note that ranking operates at the
        physical word-crop instance level: recurring vocabulary words
        written across different lines represent separate handwriting motor
        executions with distinct bounding boxes and stroke geometry.
    </p>

    {''.join(word_sections)
    if word_sections
    else
    '<p>No word-level anomaly records were available for this sample.</p>'}

</section>


<!-- ======================================================
     SECTION 9
     ====================================================== -->

<section id="gradcam">

    <div class="section-header">

        <div class="section-number">09</div>

        <div>
            <div class="eyebrow">Visual Explanation</div>
            <h2>ResNet50 and Grad-CAM Explanation</h2>
        </div>

    </div>

    <p>
        Grad-CAM highlights image regions that contribute strongly
        to the ResNet50 handwriting representation.
    </p>

    <p>
        It provides visual localization of influential image
        regions. Because the form-level ResNet50 features are cached
        and the final anomaly score is produced by LOF in the
        256-dimensional embedding space, the Grad-CAM visualization
        should not be interpreted as a direct gradient explanation
        of the LOF anomaly score.
    </p>

    <h3>
        Heatmap Interpretation
    </h3>

    <div class="legend">

        <div class="legend-item">
            <div
                class="legend-box"
                style="background:#0000ff;"
            ></div>
            Blue — lower activation
        </div>

        <div class="legend-item">
            <div
                class="legend-box"
                style="background:#00ff00;"
            ></div>
            Green — moderate activation
        </div>

        <div class="legend-item">
            <div
                class="legend-box"
                style="background:#ffff00;"
            ></div>
            Yellow — stronger activation
        </div>

        <div class="legend-item">
            <div
                class="legend-box"
                style="background:#ff0000;"
            ></div>
            Red — highest activation
        </div>

    </div>

    <p>
        The visualizations shown in this report are restricted to
        the exact sample and word IDs requested for
        <strong>{html_escape(sample_id)}</strong>.
    </p>

</section>


<!-- ======================================================
     SECTION 10
     ====================================================== -->

<section id="global-local">

    <div class="section-header">

        <div class="section-number">10</div>

        <div>
            <div class="eyebrow">Evidence Comparison</div>
            <h2>Global vs Local Evidence</h2>
        </div>

    </div>

    <p>
        The comparison below examines the relationship between the
        form-level LOF anomaly score and aggregate word-level
        anomaly measurements.
    </p>

    <p>
        The LOF pathway operates on the learned 256-dimensional
        form representation, while the word pathway uses individual
        word features and robust reference distributions.
    </p>

    {chart_section}

    <div class="evidence-grid">

        <div class="evidence-card">

            <span>LOF Percentile</span>

            <strong>
                {format_percent(lp, 1)}
            </strong>

        </div>

        <div class="evidence-card">

            <span>Word Anomaly Percentile</span>

            <strong>
                {format_percent(wp, 1)}
            </strong>

        </div>

        <div class="evidence-card">

            <span>Agreement Category</span>

            <strong>
                {html_escape(agreement)}
            </strong>

        </div>

    </div>

    <p>
        {findings["relative_position"]}
    </p>

    <div class="note">

        <div class="status-icon">i</div>

        <div>

            <strong>
                Global–Local Evidence Score
            </strong>

            <p>
                When available, this score is calculated using a
                50/50 percentile-based research heuristic combining
                the form-level LOF anomaly percentile and the
                word-level anomaly percentile.
            </p>

            <p>
                It is intended to summarize agreement between two
                evidence streams. It is not a probability, confidence
                score, clinical severity score, or diagnostic measure.
            </p>

        </div>

    </div>

</section>


<!-- ======================================================
     SECTION 11
     ====================================================== -->

<section id="interpretation">

    <div class="section-header">

        <div class="section-number">11</div>

        <div>
            <div class="eyebrow">Interpretation</div>
            <h2>Sample-Specific Findings</h2>
        </div>

    </div>

    <h3>What is unusual?</h3>

    <p>
        {findings["finding"]}
    </p>

    <h3>
        Which form feature is most LOF-sensitive?
    </h3>

    <p>
        {findings["lof_feature_pattern"]}
    </p>

    <h3>
        Where is the local evidence?
    </h3>

    <p>
        {findings["word_pattern"]}
    </p>

    <h3>
        How strong is the local evidence?
    </h3>

    <p>
        {findings["z_pattern"]}
    </p>

    <h3>
        How do the two pipelines agree?
    </h3>

    <p>
        The agreement category for this sample is
        <strong>{html_escape(agreement)}</strong>.
        This category is dataset-relative and describes agreement
        between the LOF-based form-level analysis and the word-level
        anomaly analysis rather than a clinical condition.
    </p>

    <h3>
        What this sample tells us
    </h3>

    <p>
        {findings["what_it_tells"]}
    </p>

</section>


<!-- ======================================================
     SECTION 12
     ====================================================== -->

<section id="limitations">

    <div class="section-header">

        <div class="section-number">12</div>

        <div>
            <div class="eyebrow">Research Boundaries</div>
            <h2>Limitations</h2>
        </div>

    </div>

    <ul>

        <li>
            The IAM dataset primarily provides adult handwriting
            and does not provide reliable clinical dysgraphia labels
            for this pipeline.
        </li>

        <li>
            The form-level LOF score measures local outlier behaviour
            in the learned embedding space and is not a diagnostic
            probability.
        </li>

        <li>
            The 0–100 normalized LOF score, where used, is a
            dataset-relative normalization and should not be
            interpreted as a percentage, probability, confidence,
            or severity measure.
        </li>

        <li>
            Word-level counts can differ from form-level detected
            word counts because of filtering and segmentation rules.
        </li>

        <li>
            Grad-CAM provides visual localization of influential
            ResNet50 representation regions rather than a direct
            explanation of the final LOF anomaly score.
        </li>

        <li>
            LOF feature sensitivity is a controlled
            one-feature-at-a-time perturbation analysis. It measures
            sensitivity of the LOF score under feature replacement
            and should not be interpreted as causal attribution.
        </li>

        <li>
            The LOF sensitivity percentage represents the feature's
            share of the total absolute LOF change measured across
            the nine separate perturbations. It is not a probability,
            confidence, severity, or diagnostic percentage.
        </li>

        <li>
            The sensitivity analysis does not capture interactions
            between multiple OpenCV features.
        </li>

        <li>
            The Global–Local Evidence Score is a research heuristic
            combining two percentile-based evidence streams.
        </li>

        <li>
            The anomaly framework requires evaluation on clinically
            labelled handwriting data before any clinical or
            diagnostic interpretation could be considered.
        </li>

    </ul>

</section>


<!-- ======================================================
     SECTION 13
     ====================================================== -->

<section id="conclusion">

    <div class="section-header">

        <div class="section-number">13</div>

        <div>
            <div class="eyebrow">Final Summary</div>
            <h2>Sample-Specific Conclusion</h2>
        </div>

    </div>

    <p>
        For <strong>{html_escape(sample_id)}</strong>, the analysis
        produced the following sample-specific evidence:
    </p>

    <div class="summary-grid">

        <div class="metric-card">
            <span>LOF Form-Level Anomaly</span>
            <strong>
                {format_number(lof_score, 2)}
            </strong>
        </div>

        <div class="metric-card">
            <span>Mean Local Word Anomaly</span>
            <strong>
                {format_number(mean_word, 3)}
            </strong>
        </div>

        <div class="metric-card">
            <span>Strongest Word Anomaly</span>
            <strong>
                {format_number(max_word, 3)}
            </strong>
        </div>

        <div class="metric-card">
            <span>Most LOF-Sensitive Feature</span>
            <strong>
                {html_escape(
                    findings["lof_feature_label"]
                    or "Not available"
                )}
            </strong>
        </div>

        <div class="metric-card">
            <span>LOF Feature Sensitivity</span>
            <strong>
                {format_percent(
                    findings["lof_feature_sensitivity"],
                    2,
                )}
            </strong>
        </div>

        <div class="metric-card">
            <span>Agreement</span>
            <strong>
                {html_escape(agreement)}
            </strong>
        </div>

    </div>

    <p>
        {findings["what_it_tells"]}
    </p>

    <div class="note">

        <div class="status-icon">i</div>

        <div>

            <strong>
                Overall research interpretation
            </strong>

            <p>
                This report combines whole-form handwriting
                representation, form-level LOF anomaly analysis,
                handcrafted OpenCV measurements, LOF feature
                sensitivity, local word-level anomaly evidence,
                and visual ResNet50 explanation.
            </p>

            <p>
                The system is intended as an explainable handwriting
                anomaly-analysis framework and research tool, not as
                a standalone clinical diagnostic system.
            </p>

        </div>

    </div>

</section>


<!-- ======================================================
     FOOTER
     ====================================================== -->

<footer>

    <div class="footer-title">
        Explainable Handwriting Anomaly Detection
    </div>

    <div>
        ResNet50 + OpenCV Features + Feature Fusion +
        Attention + 256-D Embedding + LOF +
        Word-Level Analysis + Grad-CAM
    </div>

    <div style="margin-top:10px;">
        Sample-specific research report —
        {html_escape(sample_id)}
    </div>

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
        description=(
            "Generate a sample-specific "
            "handwriting anomaly report."
        )
    )

    parser.add_argument(
        "pos_sample_id",
        nargs="?",
        default=None,
        help="Optional positional IAM sample ID or image filename, e.g. a01-000u or a01-000u.png",
    )

    parser.add_argument(
        "--sample",
        "--sample_id",
        dest="sample_id",
        default=None,
        help="IAM sample ID or image filename, e.g. a01-000u or a01-000u.png (defaults to a01-000u)",
    )

    args = parser.parse_args()

    raw_sample = args.sample_id or args.pos_sample_id or "a01-000u"
    sample_id = str(raw_sample).strip()
    if sample_id.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) or "/" in sample_id or "\\" in sample_id:
        sample_id = Path(sample_id).stem.strip()

    print("=" * 70)
    print(
        "EXPLAINABLE HANDWRITING "
        "ANOMALY REPORT"
    )
    print("=" * 70)

    print(
        f"Sample ID: {sample_id}"
    )

    print()

    # --------------------------------------------------------
    # Automatically generate word Grad-CAM visuals if missing
    # --------------------------------------------------------
    explain_dir = get_sample_explainability_folder(sample_id)
    if not explain_dir.exists() or len(list(explain_dir.glob("*_original.png"))) == 0:
        gradcam_script = Path(__file__).resolve().parent / "word_gradcam.py"
        if gradcam_script.exists():
            print(f"Generating word Grad-CAM visuals for '{sample_id}'...")
            subprocess.run([sys.executable, str(gradcam_script), "--sample", sample_id], check=False)

    # --------------------------------------------------------
    # Automatically generate LOF feature sensitivity if missing
    # --------------------------------------------------------
    sensitivity_csv = METADATA_DIR / f"{safe_filename(sample_id)}_lof_feature_sensitivity.csv"
    if not sensitivity_csv.exists():
        attr_script = Path(__file__).resolve().parent / "form_feature_attribution.py"
        if attr_script.exists():
            print(f"Generating LOF feature sensitivity for '{sample_id}'...")
            subprocess.run([sys.executable, str(attr_script), sample_id], check=False)

    (
        master,
        anomaly,
        word_anomaly,
        word_summary,
        metadata,
        words_metadata,
    ) = load_data()

    # --------------------------------------------------------
    # Verify sample
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
            f"ERROR: Sample '{sample_id}' "
            "was not found in the available "
            "metadata/anomaly files."
        )

        return

    # --------------------------------------------------------
    # Load LOF sensitivity ONCE
    # --------------------------------------------------------

    sensitivity_df, sensitivity_file = (
        load_lof_feature_sensitivity(
            sample_id
        )
    )

    if sensitivity_df.empty:

        print(
            "WARNING:"
        )

        print(
            "LOF feature sensitivity file was "
            "not found or could not be loaded."
        )

        print(
            "Expected:"
        )

        print(
            sensitivity_file
        )

        print()

        print(
            "The report will still be generated, "
            "but Section 5 will show a warning."
        )

    else:

        top_feature = safe_text(
            sensitivity_df.iloc[0].get(
                "feature"
            ),
            "N/A",
        )

        top_label = feature_label(
            top_feature,
            FORM_LOF_FEATURES,
        )

        # IMPORTANT:
        # CURRENT CSV COLUMN NAMES

        top_change = safe_float(
            sensitivity_df.iloc[0].get(
                "lof_score_change"
            )
        )

        top_sensitivity = safe_float(
            sensitivity_df.iloc[0].get(
                "relative_lof_sensitivity_pct"
            )
        )

        print(
            "LOF feature sensitivity loaded."
        )

        print(
            f"Most LOF-sensitive feature: "
            f"{top_label}"
        )

        print(
            f"LOF score change: "
            f"{format_number(top_change, 6)}"
        )

        print(
            f"Relative sensitivity: "
            f"{format_percent(top_sensitivity, 2)}"
        )

        print()

    # --------------------------------------------------------
    # Verify LOF
    # --------------------------------------------------------

    lof_available = (
        "lof_anomaly_score" in sample_row.index
        or "lof_raw_anomaly" in sample_row.index
        or "resnet_raw_anomaly" in sample_row.index
        or "anomaly_score" in sample_row.index
        or "resnet_anomaly_score" in sample_row.index
    )

    if not lof_available:

        print(
            "WARNING: LOF columns were not found "
            "in the selected sample record."
        )

        print(
            "Expected one of:"
        )

        print(
            "  resnet_raw_anomaly"
        )

        print(
            "  lof_anomaly_score"
        )

        print(
            "  anomaly_score"
        )

        print()

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    html_content = generate_html(
        sample_id,
        sample_row,
        master,
        word_anomaly,
        word_summary,
        metadata,
        words_metadata,
        lof_sensitivity_df=sensitivity_df,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / f"{safe_filename(sample_id)}"
        "_final_report.html"
    )

    output_file.write_text(
        html_content,
        encoding="utf-8",
    )

    print()
    print("DONE.")
    print()
    print("Report generated:")
    print(output_file)
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()