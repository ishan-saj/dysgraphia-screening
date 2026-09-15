from pathlib import Path
import argparse
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw" / "IAM" / "images"

METADATA_DIR = DATA_DIR / "metadata"

METADATA_ALL_FILE = METADATA_DIR / "metadata_all.csv"

WORD_ANOMALY_FILE = (
    METADATA_DIR / "all_samples_word_anomaly.csv"
)

WORDS_METADATA_FILE = (
    METADATA_DIR / "words_metadata.csv"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT / "best_dysgraphia_encoder.pth"
)


# ============================================================
# COMMAND LINE
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Generate sample-specific word-level "
        "Grad-CAM explainability."
    )
)

parser.add_argument(
    "--sample_id",
    "--sample",
    dest="sample_id",
    required=True,
    help="Sample ID, e.g. a01-063 or j06-051"
)

args = parser.parse_args()

SAMPLE_ID = str(args.sample_id).strip()


# ============================================================
# SAMPLE-SPECIFIC OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    METADATA_DIR /
    f"{SAMPLE_ID}_word_explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("WORD-LEVEL ANOMALY GRAD-CAM")
print("=" * 70)

print(f"Requested sample: {SAMPLE_ID}")
print()


# ============================================================
# FILE CHECKS
# ============================================================

required_files = [
    METADATA_ALL_FILE,
    WORD_ANOMALY_FILE,
    WORDS_METADATA_FILE
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ============================================================
# LOAD SAMPLE METADATA
# ============================================================

metadata_df = pd.read_csv(
    METADATA_ALL_FILE
)

metadata_df["sample_id"] = (
    metadata_df["sample_id"]
    .astype(str)
    .str.strip()
)

sample_rows = metadata_df[
    metadata_df["sample_id"] == SAMPLE_ID
].copy()


if sample_rows.empty:

    raise ValueError(
        f"Sample '{SAMPLE_ID}' was not found "
        f"in metadata_all.csv."
    )


sample_metadata = sample_rows.iloc[0]


if "image_path" not in sample_metadata.index:

    raise ValueError(
        "metadata_all.csv does not contain "
        "'image_path'."
    )


metadata_image_path = str(
    sample_metadata["image_path"]
).strip()


IMAGE_PATH = Path(
    metadata_image_path
)


if not IMAGE_PATH.is_absolute():

    IMAGE_PATH = (
        PROJECT_ROOT /
        IMAGE_PATH
    )


if not IMAGE_PATH.exists():

    # Additional safe fallbacks.
    candidates = [

        PROJECT_ROOT /
        metadata_image_path,

        DATA_DIR /
        metadata_image_path,

        RAW_DIR /
        Path(metadata_image_path).name
    ]

    found = None

    for candidate in candidates:

        if candidate.exists():

            found = candidate
            break

    if found is None:

        raise FileNotFoundError(
            f"Image for sample '{SAMPLE_ID}' "
            f"was not found.\n\n"
            f"Metadata path:\n"
            f"{metadata_image_path}"
        )

    IMAGE_PATH = found


print(f"Sample image: {IMAGE_PATH}")
print()


# ============================================================
# LOAD WORD ANOMALY DATA
# ============================================================

print("Loading word anomaly data...")


word_df = pd.read_csv(
    WORD_ANOMALY_FILE
)


if "form_id" not in word_df.columns:

    raise ValueError(
        "all_samples_word_anomaly.csv must "
        "contain 'form_id'."
    )


word_df["form_id"] = (
    word_df["form_id"]
    .astype(str)
    .str.strip()
)


# IMPORTANT:
# Only select the requested sample.
sample_word_df = word_df[
    word_df["form_id"] == SAMPLE_ID
].copy()


if sample_word_df.empty:

    raise ValueError(
        f"No word anomaly records exist "
        f"for sample '{SAMPLE_ID}'."
    )


# ============================================================
# REMOVE PUNCTUATION
# ============================================================

if "transcription" in sample_word_df.columns:

    punctuation_pattern = r"^[^\w]+$"

    sample_word_df = sample_word_df[
        ~sample_word_df["transcription"]
        .fillna("")
        .astype(str)
        .str.match(
            punctuation_pattern
        )
    ].copy()


# ============================================================
# VALID WORD SCORE CHECK
# ============================================================

sample_word_df[
    "word_anomaly_score"
] = pd.to_numeric(
    sample_word_df["word_anomaly_score"],
    errors="coerce"
)


sample_word_df = sample_word_df.dropna(
    subset=["word_anomaly_score"]
)


if sample_word_df.empty:

    raise ValueError(
        f"Sample '{SAMPLE_ID}' has no "
        f"valid word anomaly scores."
    )


# ============================================================
# SORT TOP WORDS
# ============================================================

sample_word_df = (
    sample_word_df
    .sort_values(
        "word_anomaly_score",
        ascending=False
    )
    .reset_index(drop=True)
)


TOP_N = min(
    10,
    len(sample_word_df)
)


top_words = (
    sample_word_df
    .head(TOP_N)
    .copy()
)


print(
    f"Valid words for {SAMPLE_ID}: "
    f"{len(sample_word_df)}"
)

print(
    f"Top anomalous words selected: "
    f"{TOP_N}"
)

print()


# ============================================================
# LOAD IAM WORD METADATA
# ============================================================

words_metadata = pd.read_csv(
    WORDS_METADATA_FILE
)


required_word_columns = [
    "form_id",
    "word_id",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height"
]


for column in required_word_columns:

    if column not in words_metadata.columns:

        raise ValueError(
            f"words_metadata.csv is missing "
            f"'{column}'."
        )


words_metadata["form_id"] = (
    words_metadata["form_id"]
    .astype(str)
    .str.strip()
)


words_metadata["word_id"] = (
    words_metadata["word_id"]
    .astype(str)
    .str.strip()
)


# ============================================================
# ONLY REQUESTED SAMPLE
# ============================================================

sample_boxes = words_metadata[
    words_metadata["form_id"] == SAMPLE_ID
].copy()


if sample_boxes.empty:

    raise ValueError(
        f"No word bounding boxes found "
        f"for sample '{SAMPLE_ID}'."
    )


# ============================================================
# KEEP VALID SEGMENTATION
# ============================================================

if "segmentation_status" in sample_boxes.columns:

    sample_boxes = sample_boxes[
        sample_boxes[
            "segmentation_status"
        ]
        .astype(str)
        .str.lower()
        .eq("ok")
    ].copy()


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH),
    cv2.IMREAD_COLOR
)


if image is None:

    raise ValueError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


image_height, image_width = image.shape[:2]


# ============================================================
# LOAD IMAGE-NET PRETRAINED RESNET50
# ============================================================

print(
    "Loading ImageNet-pretrained ResNet50..."
)


weights = ResNet50_Weights.DEFAULT

resnet = resnet50(
    weights=weights
)

resnet.fc = nn.Identity()

resnet.eval()

resnet.to(DEVICE)


# ============================================================
# CHECK PROJECT CHECKPOINT
# ============================================================

if CHECKPOINT_FILE.exists():

    print(
        f"Project checkpoint found:\n"
        f"{CHECKPOINT_FILE}"
    )

    print(
        "Note: the checkpoint is not loaded "
        "into ResNet because the trained pipeline "
        "uses cached/frozen ResNet features."
    )

else:

    print(
        "WARNING: Project checkpoint not found:"
    )

    print(
        CHECKPOINT_FILE
    )


print()


# ============================================================
# GRAD-CAM HOOKS
# ============================================================

activations = None

gradients = None


def forward_hook(
    module,
    input_data,
    output
):

    global activations

    activations = output


def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


target_layer = (
    resnet.layer4[-1]
)


forward_handle = (
    target_layer.register_forward_hook(
        forward_hook
    )
)


backward_handle = (
    target_layer.register_full_backward_hook(
        backward_hook
    )
)


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_crop(crop):

    resized = cv2.resize(
        crop,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    resized = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    resized = (
        resized.astype(np.float32)
        / 255.0
    )

    tensor = torch.from_numpy(
        resized
    ).permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    # ImageNet normalization.
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    tensor = (
        tensor - mean
    ) / std

    return tensor.to(DEVICE)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(crop):

    global activations
    global gradients

    activations = None
    gradients = None

    tensor = preprocess_crop(crop)

    tensor.requires_grad_(True)

    resnet.zero_grad(
        set_to_none=True
    )

    features = resnet(
        tensor
    )

    # Representation magnitude target.
    target = torch.norm(
        features,
        p=2,
        dim=1
    ).sum()

    target.backward()

    if (
        activations is None
        or gradients is None
    ):

        raise RuntimeError(
            "Grad-CAM hooks did not capture "
            "activations/gradients."
        )

    activation_map = (
        activations[0]
        .detach()
        .cpu()
        .numpy()
    )

    gradient_map = (
        gradients[0]
        .detach()
        .cpu()
        .numpy()
    )

    weights = np.mean(
        gradient_map,
        axis=(1, 2)
    )

    cam = np.zeros(
        activation_map.shape[1:],
        dtype=np.float32
    )

    for channel in range(
        activation_map.shape[0]
    ):

        cam += (
            weights[channel]
            *
            activation_map[channel]
        )

    cam = np.maximum(
        cam,
        0
    )

    if cam.max() <= 0:

        return (
            np.zeros(
                crop.shape[:2],
                dtype=np.float32
            ),
            np.zeros(
                crop.shape[:2],
                dtype=np.uint8
            )
        )

    cam = (
        cam - cam.min()
    ) / (
        cam.max()
        - cam.min()
        + 1e-8
    )

    cam = cv2.resize(
        cam,
        (
            crop.shape[1],
            crop.shape[0]
        ),
        interpolation=cv2.INTER_CUBIC
    )

    cam = cv2.GaussianBlur(
        cam,
        (0, 0),
        sigmaX=2.0
    )

    cam = (
        cam - cam.min()
    ) / (
        cam.max()
        - cam.min()
        + 1e-8
    )

    # ========================================================
    # INK MASK
    # ========================================================

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    _, ink_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    ink_mask = cv2.dilate(
        ink_mask,
        kernel,
        iterations=1
    )

    if np.count_nonzero(
        ink_mask
    ) < 10:

        ink_mask = np.full(
            crop.shape[:2],
            255,
            dtype=np.uint8
        )

    mask_float = (
        ink_mask.astype(
            np.float32
        )
        / 255.0
    )

    cam = cam * mask_float

    cam = cv2.GaussianBlur(
        cam,
        (0, 0),
        sigmaX=4.0
    )

    if cam.max() > 0:

        cam = cam / (
            cam.max()
            + 1e-8
        )

    return (
        cam,
        ink_mask
    )


# ============================================================
# LEGEND
# ============================================================

def create_legend(width):

    height = 85

    legend = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8
    )

    values = np.linspace(
        0,
        255,
        width
    ).astype(np.uint8)

    gradient = cv2.applyColorMap(
        values.reshape(1, -1),
        cv2.COLORMAP_JET
    )

    legend[
        8:32,
        :
    ] = np.repeat(
        gradient,
        24,
        axis=0
    )

    cv2.rectangle(
        legend,
        (0, 8),
        (width - 1, 31),
        (0, 0, 0),
        1
    )

    cv2.putText(
        legend,
        "Low contribution",
        (5, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        legend,
        "Moderate",
        (
            max(5, width // 2 - 35),
            55
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        legend,
        "High contribution",
        (
            max(5, width - 120),
            55
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        legend,
        (
            "Blue = lower | Red = highest "
            "representation contribution"
        ),
        (5, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (60, 60, 60),
        1,
        cv2.LINE_AA
    )

    return legend


# ============================================================
# FEATURE EXPLANATIONS
# ============================================================

FEATURE_EXPLANATIONS = {

    "aspect_ratio":
        "The word has an unusual width-to-height relationship compared with the IAM reference words.",

    "ink_aspect_ratio":
        "The handwriting ink has an unusual width-to-height relationship compared with the reference distribution.",

    "ink_density":
        "The amount of foreground handwriting ink differs substantially from the IAM reference distribution.",

    "centroid_x_ratio":
        "The horizontal center of the handwriting ink is unusually shifted within the word region.",

    "centroid_y_ratio":
        "The vertical center of the handwriting ink is unusually shifted within the word region.",

    "largest_component_ratio":
        "A dominant connected handwriting component has an unusual relative size.",

    "horizontal_projection_std":
        "The horizontal distribution of handwriting ink is unusually variable compared with the reference.",

    "vertical_projection_std":
        "The vertical distribution of handwriting ink is unusually variable compared with the reference.",

    "lower_ink_ratio":
        "The amount of handwriting ink in the lower portion of the word differs substantially from the reference."
}


# ============================================================
# EXPLANATION
# ============================================================

def explain_word(row):

    feature = str(
        row.get(
            "main_anomaly_feature",
            "unknown"
        )
    )

    try:

        z_score = float(
            row.get(
                "main_feature_z",
                0
            )
        )

    except Exception:

        z_score = 0.0

    score = float(
        row["word_anomaly_score"]
    )

    feature_explanation = (
        FEATURE_EXPLANATIONS.get(
            feature,
            "The handwriting region contains a feature that differs from the reference distribution."
        )
    )

    return (
        f"The word has a word anomaly score of "
        f"{score:.2f}. "
        f"The primary contributing feature is "
        f"{feature}, with a robust z-score of "
        f"{z_score:.2f}. "
        f"{feature_explanation} "
        f"The Grad-CAM highlights regions that "
        f"contribute strongly to the ResNet50 "
        f"handwriting representation. "
        f"This visualization is model evidence of "
        f"unusual representation patterns and is "
        f"not a clinical diagnosis of dysgraphia."
    )


# ============================================================
# PROCESS TOP WORDS
# ============================================================

report_rows = []

print(
    "Generating Grad-CAM images..."
)

print()


for rank, (_, row) in enumerate(
    top_words.iterrows(),
    start=1
):

    word = str(
        row.get(
            "transcription",
            ""
        )
    ).strip()

    word_id = str(
        row.get(
            "word_id",
            ""
        )
    ).strip()


    # ========================================================
    # EXACT WORD-ID MATCH
    # ========================================================

    box_rows = sample_boxes[
        sample_boxes["word_id"]
        == word_id
    ].copy()


    if box_rows.empty:

        print(
            f"WARNING: No bounding box for "
            f"{word_id} ({word})."
        )

        continue


    box = box_rows.iloc[0]


    try:

        x = int(
            float(box["bbox_x"])
        )

        y = int(
            float(box["bbox_y"])
        )

        w = int(
            float(box["bbox_width"])
        )

        h = int(
            float(box["bbox_height"])
        )

    except Exception:

        print(
            f"WARNING: Invalid bounding box "
            f"for {word_id}."
        )

        continue


    if w <= 0 or h <= 0:

        print(
            f"WARNING: Invalid dimensions "
            f"for {word_id}."
        )

        continue


    # ========================================================
    # SAFE CROP
    # ========================================================

    padding = 10

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image_width,
        x + w + padding
    )

    y2 = min(
        image_height,
        y + h + padding
    )

    crop = image[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:

        print(
            f"WARNING: Empty crop for "
            f"{word_id}."
        )

        continue


    # ========================================================
    # GRAD-CAM
    # ========================================================

    try:

        cam, activation_mask = (
            generate_gradcam(
                crop
            )
        )

    except Exception as error:

        print(
            f"WARNING: Grad-CAM failed "
            f"for {word_id}: {error}"
        )

        continue


    # ========================================================
    # HEATMAP
    # ========================================================

    heatmap_gray = (
        cam * 255
    ).astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        heatmap_gray,
        cv2.COLORMAP_JET
    )


    # ========================================================
    # TRANSPARENT OVERLAY
    # ========================================================

    mask_float = (
        activation_mask.astype(
            np.float32
        )
        / 255.0
    )

    alpha = (
        0.15
        +
        0.65 * cam
    )

    alpha = (
        alpha
        * mask_float
    )

    alpha = np.clip(
        alpha,
        0,
        0.80
    )

    alpha = alpha[
        ...,
        np.newaxis
    ]

    crop_float = (
        crop.astype(
            np.float32
        )
    )

    heatmap_float = (
        heatmap.astype(
            np.float32
        )
    )

    overlay = (
        crop_float
        * (1 - alpha)
        +
        heatmap_float
        * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )


    # ========================================================
    # CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        activation_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 255, 255),
        1
    )


    # ========================================================
    # ACTIVATION STATISTICS
    # ========================================================

    total_area = (
        activation_mask.shape[0]
        *
        activation_mask.shape[1]
    )

    activation_area = np.count_nonzero(
        activation_mask
    )

    activation_percentage = (
        activation_area
        /
        max(1, total_area)
        *
        100
    )


    # ========================================================
    # LEGEND
    # ========================================================

    legend = create_legend(
        overlay.shape[1]
    )

    overlay_with_legend = np.vstack(
        [
            overlay,
            legend
        ]
    )


    # ========================================================
    # SAFE FILE NAME
    # ========================================================

    safe_word = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        word
    )

    if not safe_word:

        safe_word = "word"


    base_name = (
        f"{rank:02d}_"
        f"{safe_word}_"
        f"{word_id}"
    )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    original_file = (
        OUTPUT_DIR /
        f"{base_name}_original.png"
    )

    gradcam_file = (
        OUTPUT_DIR /
        f"{base_name}_gradcam.png"
    )

    comparison_file = (
        OUTPUT_DIR /
        f"{base_name}_comparison.png"
    )


    # ========================================================
    # SAVE ORIGINAL
    # ========================================================

    cv2.imwrite(
        str(original_file),
        crop
    )


    # ========================================================
    # SAVE GRAD-CAM
    # ========================================================

    cv2.imwrite(
        str(gradcam_file),
        overlay_with_legend
    )


    # ========================================================
    # COMPARISON
    # ========================================================

    original_display = crop.copy()

    gradcam_display = (
        overlay_with_legend.copy()
    )

    target_height = (
        gradcam_display.shape[0]
    )

    original_display = cv2.resize(
        original_display,
        (
            max(
                1,
                int(
                    original_display.shape[1]
                    *
                    target_height
                    /
                    max(
                        1,
                        original_display.shape[0]
                    )
                )
            ),
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )

    comparison_body = np.hstack(
        [
            original_display,
            gradcam_display
        ]
    )

    title_height = 45

    title = np.full(
        (
            title_height,
            comparison_body.shape[1],
            3
        ),
        255,
        dtype=np.uint8
    )

    midpoint = (
        original_display.shape[1]
    )

    cv2.putText(
        title,
        "Original Word",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        title,
        "Anomaly Grad-CAM",
        (
            midpoint + 20,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    comparison = np.vstack(
        [
            title,
            comparison_body
        ]
    )

    cv2.imwrite(
        str(comparison_file),
        comparison
    )


    # ========================================================
    # TEXT EXPLANATION
    # ========================================================

    explanation = explain_word(
        row
    )


    # ========================================================
    # SAVE REPORT ROW
    # ========================================================

    report_rows.append({

        "rank":
            rank,

        "sample_id":
            SAMPLE_ID,

        "word":
            word,

        "word_id":
            word_id,

        "word_anomaly_score":
            float(
                row["word_anomaly_score"]
            ),

        "main_anomaly_feature":
            str(
                row.get(
                    "main_anomaly_feature",
                    ""
                )
            ),

        "main_feature_z":
            float(
                row.get(
                    "main_feature_z",
                    0
                )
            ),

        "bbox_x":
            x,

        "bbox_y":
            y,

        "bbox_width":
            w,

        "bbox_height":
            h,

        "high_activation_percentage":
            float(
                activation_percentage
            ),

        "original_crop":
            str(
                original_file.relative_to(
                    PROJECT_ROOT
                )
            ),

        "gradcam_image":
            str(
                gradcam_file.relative_to(
                    PROJECT_ROOT
                )
            ),

        "comparison_image":
            str(
                comparison_file.relative_to(
                    PROJECT_ROOT
                )
            ),

        "explanation":
            explanation
    })


    print(
        f"#{rank:<2} "
        f"{word:<20} "
        f"score="
        f"{float(row['word_anomaly_score']):.4f} "
        f"feature="
        f"{row.get('main_anomaly_feature', '')} "
        f"z="
        f"{float(row.get('main_feature_z', 0)):.2f} "
        f"activation="
        f"{activation_percentage:.1f}%"
    )


# ============================================================
# SAVE CSV
# ============================================================

report = pd.DataFrame(
    report_rows
)

REPORT_FILE = (
    OUTPUT_DIR /
    "word_explainability_report.csv"
)

report.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# CREATE OVERVIEW
# ============================================================

overview_path = None


if report_rows:

    overview_images = []

    for row in report_rows:

        comparison_path = (
            PROJECT_ROOT /
            row["comparison_image"]
        )

        comparison_image = cv2.imread(
            str(comparison_path)
        )

        if comparison_image is None:

            continue

        comparison_image = cv2.resize(
            comparison_image,
            (448, 224),
            interpolation=cv2.INTER_AREA
        )

        label = (
            f"#{row['rank']} "
            f"{row['word']} | "
            f"Score: "
            f"{row['word_anomaly_score']:.2f}"
        )

        cv2.rectangle(
            comparison_image,
            (0, 0),
            (448, 35),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            comparison_image,
            label[:65],
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        overview_images.append(
            comparison_image
        )


    if overview_images:

        overview = np.vstack(
            overview_images
        )

        overview_path = (
            OUTPUT_DIR /
            "top_anomalous_words_overview.png"
        )

        cv2.imwrite(
            str(overview_path),
            overview
        )


# ============================================================
# CLEANUP
# ============================================================

forward_handle.remove()

backward_handle.remove()


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 70)
print("WORD GRAD-CAM COMPLETE")
print("=" * 70)

print()

print(f"Sample: {SAMPLE_ID}")

print(
    f"Processed words: {len(report_rows)}"
)

print()

print("Output directory:")
print(OUTPUT_DIR)

print()

print("CSV report:")
print(REPORT_FILE)

if overview_path is not None:

    print()

    print("Overview image:")
    print(overview_path)

print()

print("Generated:")
print("  - Sample-specific original word crops")
print("  - Sample-specific Grad-CAM images")
print("  - Continuous Blue -> Green -> Yellow -> Red heatmaps")
print("  - Intensity-proportional overlays")
print("  - White activation contours")
print("  - Color legends")
print("  - Side-by-side comparisons")
print("  - Quantitative explanations")
print("  - Top-word overview")
print("  - Word explainability CSV")

print()

print("DONE.")