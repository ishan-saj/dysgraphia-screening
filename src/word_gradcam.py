from pathlib import Path
import argparse
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = (
    DATA_DIR
    / "raw"
    / "IAM"
    / "images"
)

METADATA_DIR = (
    DATA_DIR
    / "metadata"
)

METADATA_ALL_FILE = (
    METADATA_DIR
    / "metadata_all.csv"
)

WORD_ANOMALY_FILE = (
    METADATA_DIR
    / "all_samples_word_anomaly.csv"
)

WORDS_METADATA_FILE = (
    METADATA_DIR
    / "words_metadata.csv"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "best_dysgraphia_encoder.pth"
)


# ============================================================
# COMMAND LINE ARGUMENT
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Generate word-level anomaly Grad-CAM "
        "explainability for an IAM handwriting sample."
    )
)

parser.add_argument(
    "--sample_id",
    "--sample",
    dest="sample_id",
    required=True,
    help="Sample ID, e.g. a01-063 or p06-052"
)

args = parser.parse_args()

SAMPLE_ID = args.sample_id


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    METADATA_DIR
    / f"{SAMPLE_ID}_word_explainability"
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
print("WORD-LEVEL ANOMALY EXPLAINABILITY")
print("=" * 70)

print(
    f"Sample: {SAMPLE_ID}"
)

print()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not METADATA_ALL_FILE.exists():

    raise FileNotFoundError(
        "metadata_all.csv not found:\n"
        f"{METADATA_ALL_FILE}"
    )


if not WORD_ANOMALY_FILE.exists():

    raise FileNotFoundError(
        "Word anomaly file not found:\n"
        f"{WORD_ANOMALY_FILE}"
    )


if not WORDS_METADATA_FILE.exists():

    raise FileNotFoundError(
        "words_metadata.csv not found:\n"
        f"{WORDS_METADATA_FILE}"
    )


# ============================================================
# LOAD SAMPLE IMAGE PATH
# FROM metadata_all.csv
# ============================================================

metadata_df = pd.read_csv(
    METADATA_ALL_FILE
)


if "sample_id" not in metadata_df.columns:

    raise ValueError(
        "metadata_all.csv must contain "
        "'sample_id'."
    )


if "image_path" not in metadata_df.columns:

    raise ValueError(
        "metadata_all.csv must contain "
        "'image_path'."
    )


sample_rows = metadata_df[
    metadata_df["sample_id"]
    .astype(str)
    == SAMPLE_ID
]


if sample_rows.empty:

    raise ValueError(
        f"No metadata found for sample "
        f"'{SAMPLE_ID}'."
    )


metadata_image_path = str(
    sample_rows.iloc[0]["image_path"]
).strip()


IMAGE_PATH = Path(
    metadata_image_path
)


# ------------------------------------------------------------
# Convert relative metadata path to project path
# ------------------------------------------------------------

if not IMAGE_PATH.is_absolute():

    IMAGE_PATH = (
        PROJECT_ROOT
        / IMAGE_PATH
    )


if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        "Image specified by metadata "
        "was not found:\n"
        f"{IMAGE_PATH}\n\n"
        f"Metadata path:\n"
        f"{metadata_image_path}"
    )


print(
    f"Image: {IMAGE_PATH}"
)

print()


# ============================================================
# LOAD WORD ANOMALY DATA
# ============================================================

print(
    "Loading word anomaly results..."
)


all_word_anomalies = pd.read_csv(
    WORD_ANOMALY_FILE
)


if "form_id" not in all_word_anomalies.columns:

    raise ValueError(
        "all_samples_word_anomaly.csv "
        "must contain 'form_id'."
    )


all_word_anomalies["form_id"] = (
    all_word_anomalies["form_id"]
    .astype(str)
)


word_anomalies = all_word_anomalies[
    all_word_anomalies["form_id"]
    == SAMPLE_ID
].copy()


if word_anomalies.empty:

    raise ValueError(
        f"No word anomaly records found "
        f"for sample '{SAMPLE_ID}'."
    )


# ============================================================
# REMOVE PUNCTUATION
# ============================================================

punctuation_pattern = (
    r"^[^\w]+$"
)


if "transcription" in word_anomalies.columns:

    word_anomalies = word_anomalies[
        ~word_anomalies[
            "transcription"
        ]
        .fillna("")
        .astype(str)
        .str.match(
            punctuation_pattern
        )
    ].copy()


# ============================================================
# SORT BY ANOMALY SCORE
# ============================================================

word_anomalies = (
    word_anomalies
    .sort_values(
        "word_anomaly_score",
        ascending=False
    )
    .reset_index(drop=True)
)


TOP_N = min(
    10,
    len(word_anomalies)
)


top_words = (
    word_anomalies
    .head(TOP_N)
    .copy()
)


print(
    f"Found {len(word_anomalies)} "
    f"handwriting words."
)

print(
    f"Selecting top {TOP_N} "
    f"anomalous words."
)

print()


# ============================================================
# PRINT TOP WORDS
# ============================================================

print(
    "TOP ANOMALOUS WORDS"
)

print(
    "-" * 70
)


for rank, (_, row) in enumerate(
    top_words.iterrows(),
    start=1
):

    word = str(
        row["transcription"]
    )

    score = float(
        row["word_anomaly_score"]
    )

    feature = str(
        row.get(
            "main_anomaly_feature",
            "unknown"
        )
    )

    z_score = float(
        row.get(
            "main_feature_z",
            0
        )
    )

    print(
        f"#{rank:<2} "
        f"{word:<20} "
        f"score={score:.4f} "
        f"feature={feature} "
        f"z={z_score:.2f}"
    )


print()


# ============================================================
# LOAD IAM WORD METADATA
# ============================================================

words_metadata = pd.read_csv(
    WORDS_METADATA_FILE
)


if "form_id" not in words_metadata.columns:

    raise ValueError(
        "words_metadata.csv must contain "
        "'form_id'."
    )


if "word_id" not in words_metadata.columns:

    raise ValueError(
        "words_metadata.csv must contain "
        "'word_id'."
    )


words_metadata["form_id"] = (
    words_metadata["form_id"]
    .astype(str)
)


sample_boxes = words_metadata[
    words_metadata["form_id"]
    == SAMPLE_ID
].copy()


if sample_boxes.empty:

    raise ValueError(
        f"No IAM word metadata found "
        f"for sample '{SAMPLE_ID}'."
    )


# ============================================================
# KEEP VALID IAM SEGMENTATION
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
        f"Could not read image:\n"
        f"{IMAGE_PATH}"
    )


image_height, image_width = (
    image.shape[:2]
)


# ============================================================
# LOAD RESNET50
# ============================================================

print(
    "Loading ResNet50..."
)


# IMPORTANT:
#
# Keep weights=None here to preserve the same
# Grad-CAM visualization method used in your
# original a01-063 implementation.
#
# The Grad-CAM is being used as a localization
# visualization of the ResNet representation,
# not as an ImageNet classification explanation.
#

resnet = resnet50(
    weights=None
)


resnet.fc = nn.Identity()


resnet.eval()

resnet.to(
    DEVICE
)


# ============================================================
# CHECKPOINT
# ============================================================

# The anomaly model checkpoint contains the fusion/
# encoder layers used during cached-feature training.
#
# The Grad-CAM visualization below specifically hooks
# the ResNet representation. Therefore the checkpoint
# is not loaded into this standalone ResNet object.
#
# We only check that the project checkpoint exists,
# so the project setup remains consistent.

if not CHECKPOINT_FILE.exists():

    print(
        "WARNING: Model checkpoint not found:"
    )

    print(
        CHECKPOINT_FILE
    )

    print(
        "Continuing because this Grad-CAM "
        "uses the standalone ResNet representation."
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
# PREPROCESS WORD CROP
# ============================================================

def preprocess_crop(crop):
    """
    Convert word crop to a 224x224 RGB tensor.

    This intentionally follows the same preprocessing
    used by the original working Grad-CAM version.
    """

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
        resized.astype(
            np.float32
        )
        / 255.0
    )


    tensor = torch.from_numpy(
        resized
    ).permute(
        2,
        0,
        1
    )


    tensor = tensor.unsqueeze(
        0
    )


    return tensor.to(
        DEVICE
    )


# ============================================================
# GENERATE REFINED GRAD-CAM
# ============================================================

def generate_gradcam(crop):
    """
    Generate refined continuous Grad-CAM.

    Returns:

        cam
            Continuous Grad-CAM intensity
            from 0 to 1.

        activation_mask
            Cleaned mask identifying
            meaningful activation regions.
    """

    global activations

    global gradients


    activations = None

    gradients = None


    tensor = preprocess_crop(
        crop
    )


    tensor.requires_grad_(True)


    resnet.zero_grad(
        set_to_none=True
    )


    features = resnet(
        tensor
    )


    # ========================================================
    # ANOMALY REPRESENTATION TARGET
    # ========================================================
    #
    # This is the same target used in the original
    # working version.
    #
    # We use the L2 magnitude of the ResNet representation.
    #
    # ========================================================

    target = torch.norm(
        features,
        p=2,
        dim=1
    ).sum()


    target.backward()


    # ========================================================
    # CHECK HOOKS
    # ========================================================

    if (
        activations is None
        or gradients is None
    ):

        raise RuntimeError(
            "Grad-CAM hooks did not capture "
            "activations or gradients."
        )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

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


    # ========================================================
    # GRADIENT WEIGHTS
    # ========================================================

    weights = np.mean(
        gradient_map,
        axis=(1, 2)
    )


    # ========================================================
    # WEIGHTED ACTIVATION
    # ========================================================

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


    # ========================================================
    # POSITIVE ACTIVATION ONLY
    # ========================================================

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


    # ========================================================
    # NORMALIZE
    # ========================================================

    cam = (
        cam - cam.min()
    ) / (
        cam.max()
        - cam.min()
        + 1e-8
    )


    # ========================================================
    # UPSAMPLE TO ORIGINAL CROP SIZE
    # ========================================================

    cam = cv2.resize(
        cam,
        (
            crop.shape[1],
            crop.shape[0]
        ),
        interpolation=cv2.INTER_CUBIC
    )


    # ========================================================
    # SMOOTH
    # ========================================================

    cam = cv2.GaussianBlur(
        cam,
        (0, 0),
        sigmaX=2.0
    )


    # ========================================================
    # NORMALIZE AGAIN
    # ========================================================

    cam = (
        cam - cam.min()
    ) / (
        cam.max()
        - cam.min()
        + 1e-8
    )


    # ========================================================
    # RESTRICT TO WORD / INK REGION
    # ========================================================
    #
    # Rather than thresholding the CAM itself (which collapses
    # the color range and produces a flat red blob), segment
    # the actual handwriting ink from the crop and use that as
    # a spatial mask. The CAM stays fully continuous inside
    # this region, which is what preserves the
    # Blue -> Green -> Yellow -> Red gradient.
    #
    # ========================================================

    gray_crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )


    _, ink_mask = cv2.threshold(
        gray_crop,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )


    # ------------------------------------------------------
    # Dilate so the heatmap covers full stroke width and a
    # small margin around each stroke, instead of only the
    # exact ink pixels.
    # ------------------------------------------------------

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )


    ink_mask = cv2.dilate(
        ink_mask,
        dilate_kernel,
        iterations=1
    )


    # ------------------------------------------------------
    # Fallback: if Otsu finds almost no ink (e.g. a very
    # faint or blank crop), fall back to the full crop so we
    # never return an empty mask.
    # ------------------------------------------------------

    if np.count_nonzero(ink_mask) < 10:

        ink_mask = np.full(
            crop.shape[:2],
            255,
            dtype=np.uint8
        )


    ink_mask_float = (
        ink_mask.astype(
            np.float32
        )
        / 255.0
    )


    cam = cam * ink_mask_float


    # ========================================================
    # GAUSSIAN SMOOTHING
    # ========================================================
    #
    # A wider blur than the initial smoothing pass, so the
    # activation diffuses smoothly across and between nearby
    # strokes rather than sitting in isolated hard-edged
    # pixels.
    #
    # ========================================================

    cam = cv2.GaussianBlur(
        cam,
        (0, 0),
        sigmaX=4.0
    )


    # ========================================================
    # RE-NORMALIZE WITHIN THE INK REGION
    # ========================================================

    if cam.max() > 0:

        cam = cam / (
            cam.max()
            + 1e-8
        )


    activation_mask = (
        ink_mask
    )


    return (
        cam,
        activation_mask
    )


# ============================================================
# CREATE COLOR LEGEND
# ============================================================

def create_legend(width):

    legend_height = 85


    legend = np.full(
        (
            legend_height,
            width,
            3
        ),
        255,
        dtype=np.uint8
    )


    # ========================================================
    # CONTINUOUS COLOR GRADIENT
    # ========================================================

    gradient_values = np.linspace(
        0,
        255,
        width
    ).astype(
        np.uint8
    )


    gradient_bar = cv2.applyColorMap(
        gradient_values.reshape(
            1,
            -1
        ),
        cv2.COLORMAP_JET
    )


    legend[
        8:32,
        :
    ] = np.repeat(
        gradient_bar,
        24,
        axis=0
    )


    # ========================================================
    # BORDER
    # ========================================================

    cv2.rectangle(
        legend,
        (0, 8),
        (width - 1, 31),
        (0, 0, 0),
        1
    )


    # ========================================================
    # LABELS
    # ========================================================

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
            max(
                5,
                width // 2 - 35
            ),
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
            max(
                5,
                width - 120
            ),
            55
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # EXPLANATION
    # ========================================================

    cv2.putText(
        legend,
        (
            "Blue = lower model contribution | "
            "Red = highest model contribution"
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
        (
            "The word has an unusual width-to-height "
            "relationship compared with the IAM "
            "reference words."
        ),

    "ink_aspect_ratio":
        (
            "The handwriting ink has an unusual "
            "width-to-height relationship compared "
            "with the reference distribution."
        ),

    "ink_density":
        (
            "The amount of foreground handwriting "
            "ink inside the word region differs "
            "substantially from the IAM reference "
            "distribution."
        ),

    "centroid_x_ratio":
        (
            "The horizontal center of the handwriting "
            "ink is unusually shifted within the "
            "word region."
        ),

    "centroid_y_ratio":
        (
            "The vertical center of the handwriting "
            "ink is unusually shifted within the "
            "word region."
        ),

    "largest_component_ratio":
        (
            "A dominant connected handwriting component "
            "has an unusual relative size."
        ),

    "horizontal_projection_std":
        (
            "The horizontal distribution of handwriting "
            "ink is unusually variable compared with "
            "the reference."
        ),

    "vertical_projection_std":
        (
            "The vertical distribution of handwriting "
            "ink is unusually variable compared with "
            "the reference."
        ),

    "lower_ink_ratio":
        (
            "The amount of handwriting ink in the lower "
            "portion of the word differs substantially "
            "from the reference."
        )
}


# ============================================================
# EXPLAIN WORD
# ============================================================

def explain_word(row):

    feature = str(
        row.get(
            "main_anomaly_feature",
            "unknown"
        )
    )


    z_score = float(
        row.get(
            "main_feature_z",
            0
        )
    )


    score = float(
        row["word_anomaly_score"]
    )


    feature_explanation = (
        FEATURE_EXPLANATIONS.get(
            feature,
            (
                "The handwriting region contains "
                "a feature that differs substantially "
                "from the reference distribution."
            )
        )
    )


    return (
        f"The word has an anomaly score of "
        f"{score:.2f}. "
        f"The primary contributing feature is "
        f"{feature}, with a robust z-score of "
        f"{z_score:.2f}. "
        f"{feature_explanation} "
        f"The Grad-CAM highlights image regions "
        f"that contribute most strongly to the "
        f"ResNet image representation used during "
        f"the anomaly analysis. "
        f"These highlighted regions should be "
        f"interpreted as model evidence for "
        f"unusualness, not as a clinical diagnosis "
        f"of dysgraphia."
    )


# ============================================================
# PROCESS TOP WORDS
# ============================================================

report_rows = []


print(
    "Generating word-level explanations..."
)

print()


for rank, (_, row) in enumerate(
    top_words.iterrows(),
    start=1
):

    word = str(
        row["transcription"]
    )


    word_id = str(
        row.get(
            "word_id",
            ""
        )
    )


    # ========================================================
    # FIND IAM BOUNDING BOX
    # ========================================================

    box_rows = sample_boxes[
        sample_boxes["word_id"]
        .astype(str)
        == word_id
    ]


    if box_rows.empty:

        print(
            f"WARNING: Bounding box not found "
            f"for {word_id}. Skipping."
        )

        continue


    box = box_rows.iloc[0]


    x = int(
        box["bbox_x"]
    )

    y = int(
        box["bbox_y"]
    )

    w = int(
        box["bbox_width"]
    )

    h = int(
        box["bbox_height"]
    )


    # ========================================================
    # PADDING
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
            f"{word}. Skipping."
        )

        continue


    # ========================================================
    # GENERATE GRAD-CAM
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
            f"for {word}: {error}"
        )

        continue


    # ========================================================
    # CONTINUOUS HEATMAP OVER THE WORD/INK REGION
    # ========================================================
    #
    # cam is continuous (0-1) across the ink region, so the
    # JET colormap renders the full
    # Blue -> Green -> Yellow -> Red gradient rather than a
    # single flat color.
    #
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
    # INTENSITY-PROPORTIONAL TRANSPARENT OVERLAY
    # ========================================================
    #
    # Transparency follows the actual continuous CAM value at
    # each pixel, so low activation fades toward the original
    # image and high activation stands out strongly. This is
    # what makes the overlay look smooth and spatially
    # distributed rather than a solid block of color.
    #
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
        0.65
        * cam
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
        *
        (1 - alpha)
        +
        heatmap_float
        *
        alpha
    )


    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )


    # ========================================================
    # WHITE ACTIVATION CONTOURS
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


    high_activation_area = (
        np.count_nonzero(
            activation_mask
        )
    )


    high_activation_percentage = (
        high_activation_area
        /
        max(
            1,
            total_area
        )
        *
        100
    )


    # ========================================================
    # ADD LEGEND
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


    base_name = (
        f"{rank:02d}_"
        f"{safe_word}_"
        f"{word_id}"
    )


    # ========================================================
    # OUTPUT FILES
    # ========================================================
    #
    # IMPORTANT:
    #
    # Files are stored directly inside
    # SAMPLE_word_explainability
    # to match your a01-063 structure.
    #
    # ========================================================

    original_file = (
        OUTPUT_DIR
        /
        f"{base_name}_original.png"
    )


    gradcam_file = (
        OUTPUT_DIR
        /
        f"{base_name}_gradcam.png"
    )


    comparison_file = (
        OUTPUT_DIR
        /
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
    # SIDE-BY-SIDE COMPARISON
    # ========================================================

    original_display = (
        crop.copy()
    )


    gradcam_display = (
        overlay_with_legend.copy()
    )


    target_height = (
        gradcam_display.shape[0]
    )


    # Resize original to same total height
    original_display = cv2.resize(
        original_display,
        (
            int(
                original_display.shape[1]
                *
                target_height
                /
                original_display.shape[0]
            ),
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )


    comparison = np.hstack(
        [
            original_display,
            gradcam_display
        ]
    )


    # ========================================================
    # TITLES
    # ========================================================

    title_height = 45


    title = np.full(
        (
            title_height,
            comparison.shape[1],
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
            comparison
        ]
    )


    # ========================================================
    # SAVE COMPARISON
    # ========================================================

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
    # ADD REPORT ROW
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
                high_activation_percentage
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


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
        f"#{rank:<2} "
        f"{word:<20} "
        f"score="
        f"{row['word_anomaly_score']:.4f} "
        f"feature="
        f"{row.get('main_anomaly_feature', '')} "
        f"z="
        f"{float(row.get('main_feature_z', 0)):.2f} "
        f"activation="
        f"{high_activation_percentage:.1f}%"
    )


# ============================================================
# SAVE WORD EXPLAINABILITY CSV
# ============================================================

report = pd.DataFrame(
    report_rows
)


REPORT_FILE = (
    OUTPUT_DIR
    /
    "word_explainability_report.csv"
)


report.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# CREATE OVERVIEW IMAGE
# ============================================================

if report_rows:

    overview_images = []


    for row in report_rows:

        comparison_image_path = (
            PROJECT_ROOT
            /
            row["comparison_image"]
        )


        comparison_image = cv2.imread(
            str(comparison_image_path)
        )


        if comparison_image is None:

            continue


        # ----------------------------------------------------
        # Resize each comparison for overview
        # ----------------------------------------------------

        comparison_image = cv2.resize(
            comparison_image,
            (448, 224),
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"#{row['rank']} "
            f"{row['word']} "
            f"| Score: "
            f"{row['word_anomaly_score']:.2f}"
        )


        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        cv2.rectangle(
            comparison_image,
            (0, 0),
            (448, 35),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            comparison_image,
            label,
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
            OUTPUT_DIR
            /
            "top_anomalous_words_overview.png"
        )


        cv2.imwrite(
            str(overview_path),
            overview
        )


    else:

        overview_path = None


else:

    overview_path = None


print()


# ============================================================
# CLEAN UP HOOKS
# ============================================================

forward_handle.remove()

backward_handle.remove()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("=" * 70)
print("WORD GRAD-CAM COMPLETE")
print("=" * 70)

print()

print(
    f"Sample: {SAMPLE_ID}"
)

print(
    f"Processed words: "
    f"{len(report_rows)}"
)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "CSV report:"
)

print(
    REPORT_FILE
)

if overview_path is not None:

    print()

    print(
        "Overview image:"
    )

    print(
        overview_path
    )

print()

print(
    "Generated:"
)

print(
    "  - Original word crops"
)

print(
    "  - Continuous Grad-CAM images"
)

print(
    "  - Blue -> Green -> Yellow -> Red heatmaps"
)

print(
    "  - Actual Grad-CAM intensity transparency"
)

print(
    "  - White activation contours"
)

print(
    "  - Color legends"
)

print(
    "  - Side-by-side comparisons"
)

print(
    "  - Quantitative explanations"
)

print(
    "  - Overview image"
)

print(
    "  - CSV report"
)

print()

print("DONE.")