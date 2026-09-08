# ============================================================
# REFINED ANOMALY + WORD-GUIDED + INK-CONSTRAINED GRAD-CAM
# ============================================================

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from model import DysgraphiaModel


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_ID = "a01-063"

IMAGE_ROOT = (
    BASE_DIR
    / "data"
    / "raw"
    / "IAM"
    / "images"
)

METADATA_DIR = (
    BASE_DIR
    / "data"
    / "metadata"
)

WORDS_METADATA = (
    METADATA_DIR
    / "words_metadata.csv"
)

FEATURE_CACHE = (
    METADATA_DIR
    / "resnet_features.pt"
)

CHECKPOINT = (
    BASE_DIR
    / "best_dysgraphia_encoder.pth"
)

OUTPUT_FILE = (
    METADATA_DIR
    / f"{SAMPLE_ID}_refined_gradcam.png"
)

OUTPUT_MASK = (
    METADATA_DIR
    / f"{SAMPLE_ID}_refined_activation_mask.png"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

# Number of anomalous words to show
TOP_WORDS = 8

# Percentile used to determine strong Grad-CAM activation
CAM_PERCENTILE = 80

# Minimum connected component size
MIN_COMPONENT_AREA = 25

# Gaussian smoothing
GAUSSIAN_SIGMA = 2.5


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(sample_id):

    matches = list(
        IMAGE_ROOT.rglob(
            f"{sample_id}.png"
        )
    )

    if not matches:

        raise FileNotFoundError(
            f"Image not found for {sample_id}"
        )

    return matches[0]


image_path = find_image(
    SAMPLE_ID
)


print("=" * 75)
print(
    f"REFINED ANOMALY GRAD-CAM: {SAMPLE_ID}"
)
print("=" * 75)

print()
print(
    "Image:",
    image_path
)

# ============================================================
# LOAD ALL-SAMPLE WORD ANOMALIES
# ============================================================

word_anomaly_file = (
    METADATA_DIR
    / "all_samples_word_anomaly.csv"
)

if not word_anomaly_file.exists():

    raise FileNotFoundError(
        "All-sample word anomaly file not found:\n"
        f"{word_anomaly_file}"
    )


print()
print(
    "Loading all-sample word anomaly results..."
)

all_word_anomalies = pd.read_csv(
    word_anomaly_file
)


# Keep only the requested sample
word_anomalies = all_word_anomalies[
    all_word_anomalies["form_id"].astype(str)
    == SAMPLE_ID
].copy()


if word_anomalies.empty:

    raise ValueError(
        f"No word anomaly records found for "
        f"{SAMPLE_ID}"
    )


print(
    "Word anomaly regions for",
    SAMPLE_ID,
    ":",
    len(word_anomalies)
)

# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------

required_word_columns = [
    "word_id",
    "transcription",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "word_anomaly_score",
]


for column in required_word_columns:

    if column not in word_anomalies.columns:

        raise ValueError(
            f"Missing column in word anomaly CSV: "
            f"{column}"
        )


# ============================================================
# SELECT TOP ANOMALOUS WORDS
# ============================================================

word_anomalies = word_anomalies.copy()

word_anomalies = word_anomalies.sort_values(
    "word_anomaly_score",
    ascending=False
)

top_words = word_anomalies.head(
    TOP_WORDS
).copy()


print()
print("Top anomalous handwriting words:")

for rank, (_, row) in enumerate(
    top_words.iterrows(),
    start=1
):

    print(
        f"{rank:2d}. "
        f"{row['transcription']} "
        f"-> "
        f"{row['word_anomaly_score']:.4f}"
    )


# ============================================================
# LOAD FEATURE CACHE
# ============================================================

print()
print("Loading cached features...")

cache = torch.load(
    FEATURE_CACHE,
    map_location="cpu",
    weights_only=False
)

cached_ids = cache["sample_ids"]

if isinstance(
    cached_ids,
    torch.Tensor
):

    cached_ids = cached_ids.tolist()

cached_ids = [
    str(x)
    for x in cached_ids
]

if SAMPLE_ID not in cached_ids:

    raise ValueError(
        f"{SAMPLE_ID} not found "
        "in feature cache."
    )

sample_index = cached_ids.index(
    SAMPLE_ID
)

cached_image_features = (
    cache["image_features"]
    .float()
)

cached_opencv_features = (
    cache["opencv_features"]
    .float()
)


print(
    "Image feature shape:",
    tuple(
        cached_image_features.shape
    )
)

print(
    "OpenCV feature shape:",
    tuple(
        cached_opencv_features.shape
    )
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading model...")

model = DysgraphiaModel(
    pretrained=True,
    freeze_backbone=False
)

model = model.to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if CHECKPOINT.exists():

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    # The cached training did not train
    # the ResNet backbone.
    #
    # Therefore keep pretrained ResNet
    # weights and load the trained fusion
    # layers.

    filtered_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "resnet."
        ):

            continue

        filtered_state_dict[
            key
        ] = value

    missing, unexpected = (
        model.load_state_dict(
            filtered_state_dict,
            strict=False
        )
    )

    print(
        "Trained fusion/attention layers loaded."
    )

    print(
        "Missing keys:",
        len(missing)
    )

    print(
        "Unexpected keys:",
        len(unexpected)
    )

else:

    print(
        "WARNING: checkpoint not found."
    )


model.eval()


# ============================================================
# RECREATE TRAIN NORMALIZATION
# ============================================================

print()
print(
    "Recreating training normalization..."
)

num_samples = len(
    cached_ids
)

generator = torch.Generator()

generator.manual_seed(
    42
)

indices = torch.randperm(
    num_samples,
    generator=generator
)

train_size = int(
    0.8 * num_samples
)

train_indices = indices[
    :train_size
]

train_opencv = (
    cached_opencv_features[
        train_indices
    ]
)

feature_mean = (
    train_opencv.mean(
        dim=0,
        keepdim=True
    )
)

feature_std = (
    train_opencv.std(
        dim=0,
        keepdim=True
    )
)

feature_std[
    feature_std < 1e-8
] = 1.0


# ============================================================
# PREPARE SAMPLE OPENCV FEATURES
# ============================================================

sample_opencv = (
    cached_opencv_features[
        sample_index
    ]
    .unsqueeze(0)
)

sample_opencv = (
    sample_opencv
    - feature_mean
) / feature_std

sample_opencv = sample_opencv.to(
    device
)


# ============================================================
# CREATE REFERENCE EMBEDDING
# ============================================================

print()
print(
    "Creating IAM reference embedding..."
)

with torch.no_grad():

    normalized_all_opencv = (
        cached_opencv_features
        - feature_mean
    ) / feature_std

    encoded_all_opencv = (
        model.opencv_encoder(
            normalized_all_opencv
        )
    )

    fused_reference = torch.cat(
        [
            cached_image_features,
            encoded_all_opencv
        ],
        dim=1
    )

    reference_embeddings, _ = (
        model.attention(
            fused_reference
        )
    )

    reference_embedding = (
        reference_embeddings.mean(
            dim=0,
            keepdim=True
        )
    )


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

original = cv2.imread(
    str(image_path)
)

if original is None:

    raise ValueError(
        "Could not read image."
    )

original_rgb = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

original_height, original_width = (
    original_rgb.shape[:2]
)


# Resize
image_resized = cv2.resize(
    original_rgb,
    (
        IMAGE_SIZE,
        IMAGE_SIZE
    )
)


# ============================================================
# IMAGE TENSOR
# ============================================================

image_tensor = torch.tensor(
    image_resized,
    dtype=torch.float32
)

image_tensor = (
    image_tensor
    .permute(2, 0, 1)
)

image_tensor = (
    image_tensor / 255.0
)

image_tensor = (
    image_tensor
    .unsqueeze(0)
)


# ImageNet normalization
mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(
    1, 3, 1, 1
)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(
    1, 3, 1, 1
)

image_tensor = (
    image_tensor - mean
) / std

image_tensor = image_tensor.to(
    device
)

image_tensor.requires_grad_(True)


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = []
gradients = []


# ============================================================
# HOOKS
# ============================================================

def forward_hook(
    module,
    input,
    output
):

    activations.clear()

    activations.append(
        output
    )


def backward_hook(
    module,
    grad_input,
    grad_output
):

    gradients.clear()

    gradients.append(
        grad_output[0]
    )


# Final ResNet convolutional block
target_layer = (
    model.resnet.layer4[-1]
)

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# FORWARD PASS
# ============================================================

print()
print(
    "Running anomaly-targeted model..."
)

image_features = model.resnet(
    image_tensor
)

encoded_opencv = (
    model.opencv_encoder(
        sample_opencv
    )
)

fused_features = torch.cat(
    [
        image_features,
        encoded_opencv
    ],
    dim=1
)

embedding, attention_weights = (
    model.attention(
        fused_features
    )
)


# ============================================================
# PROJECT-SPECIFIC ANOMALY TARGET
# ============================================================

anomaly_distance = torch.norm(
    embedding
    - reference_embedding,
    p=2,
    dim=1
)

print()
print(
    "Anomaly distance:",
    round(
        float(
            anomaly_distance.item()
        ),
        6
    )
)


# ============================================================
# BACKPROPAGATE
# ============================================================

model.zero_grad()

if image_tensor.grad is not None:

    image_tensor.grad.zero_()

print(
    "Backpropagating anomaly..."
)

anomaly_distance.backward()


# ============================================================
# READ ACTIVATIONS / GRADIENTS
# ============================================================

if not activations:

    raise RuntimeError(
        "Grad-CAM activation missing."
    )

if not gradients:

    raise RuntimeError(
        "Grad-CAM gradient missing."
    )


activation = (
    activations[0][0]
)

gradient = (
    gradients[0][0]
)


# ============================================================
# GRAD-CAM CHANNEL WEIGHTS
# ============================================================

weights = gradient.mean(
    dim=(1, 2)
)


# ============================================================
# BUILD CAM
# ============================================================

cam = torch.zeros(
    activation.shape[1:],
    dtype=torch.float32
)

for channel in range(
    activation.shape[0]
):

    cam += (
        weights[channel]
        * activation[channel]
    )


# Positive anomaly contribution
cam = F.relu(
    cam
)

cam = (
    cam
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# NORMALIZE RAW CAM
# ============================================================

cam -= cam.min()

if cam.max() > 1e-8:

    cam /= cam.max()


# ============================================================
# UPSAMPLE
# ============================================================

cam = cv2.resize(
    cam,
    (
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    interpolation=cv2.INTER_CUBIC
)


# ============================================================
# SMOOTH
# ============================================================

cam = cv2.GaussianBlur(
    cam,
    (0, 0),
    GAUSSIAN_SIGMA
)


if cam.max() > 1e-8:

    cam /= cam.max()


# ============================================================
# CREATE INK MASK
# ============================================================

gray = cv2.cvtColor(
    image_resized,
    cv2.COLOR_RGB2GRAY
)

_, ink_mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV
    + cv2.THRESH_OTSU
)

ink_binary = (
    ink_mask > 0
).astype(np.uint8)


# ============================================================
# CREATE IAM WORD MASK
# ============================================================

word_mask = np.zeros(
    (
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    dtype=np.uint8
)


scale_x = (
    IMAGE_SIZE
    / original_width
)

scale_y = (
    IMAGE_SIZE
    / original_height
)


for _, row in word_anomalies.iterrows():

    x1 = int(
        row["bbox_x"]
        * scale_x
    )

    y1 = int(
        row["bbox_y"]
        * scale_y
    )

    x2 = int(
        (
            row["bbox_x"]
            + row["bbox_width"]
        )
        * scale_x
    )

    y2 = int(
        (
            row["bbox_y"]
            + row["bbox_height"]
        )
        * scale_y
    )

    x1 = max(
        0,
        min(
            IMAGE_SIZE - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            IMAGE_SIZE - 1,
            y1
        )
    )

    x2 = max(
        0,
        min(
            IMAGE_SIZE,
            x2
        )
    )

    y2 = max(
        0,
        min(
            IMAGE_SIZE,
            y2
        )
    )

    if x2 > x1 and y2 > y1:

        word_mask[
            y1:y2,
            x1:x2
        ] = 1


# ============================================================
# KEEP ONLY HANDWRITING
# ============================================================

constraint_mask = (
    word_mask
    * ink_binary
)


# ============================================================
# LIMIT CAM TO WORD REGIONS
# ============================================================

cam = (
    cam
    * constraint_mask
)


# ============================================================
# NORMALIZE INSIDE VALID AREA
# ============================================================

valid_values = cam[
    constraint_mask > 0
]

if len(valid_values) > 0:

    low = np.percentile(
        valid_values,
        5
    )

    high = np.percentile(
        valid_values,
        99
    )

    if high > low:

        cam = (
            cam - low
        ) / (
            high - low
        )

    cam = np.clip(
        cam,
        0,
        1
    )


# ============================================================
# FIND STRONG ACTIVATION
# ============================================================

valid_values = cam[
    constraint_mask > 0
]

if len(valid_values) == 0:

    raise RuntimeError(
        "No valid handwriting activation."
    )


activation_threshold = np.percentile(
    valid_values,
    CAM_PERCENTILE
)

strong_activation = (
    cam >= activation_threshold
).astype(np.uint8)


# ============================================================
# CONNECT NEARBY ACTIVATION
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

strong_activation = cv2.morphologyEx(
    strong_activation,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# REMOVE TINY COMPONENTS
# ============================================================

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        strong_activation,
        connectivity=8
    )
)

clean_activation = np.zeros_like(
    strong_activation
)


for label in range(
    1,
    num_labels
):

    area = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    if area >= MIN_COMPONENT_AREA:

        clean_activation[
            labels == label
        ] = 1


# ============================================================
# FINAL ACTIVATION
# ============================================================

final_cam = (
    cam
    * clean_activation
)


# ============================================================
# CREATE SOFT REGION
# ============================================================

soft_cam = cv2.GaussianBlur(
    final_cam.astype(
        np.float32
    ),
    (0, 0),
    2.0
)

if soft_cam.max() > 1e-8:

    soft_cam /= soft_cam.max()


# ============================================================
# SAVE ACTIVATION MASK
# ============================================================

cv2.imwrite(
    str(OUTPUT_MASK),
    (
        clean_activation
        * 255
    ).astype(np.uint8)
)


# ============================================================
# CREATE FINAL VISUALIZATION
# ============================================================

fig = plt.figure(
    figsize=(20, 7)
)


# ============================================================
# PANEL 1
# ORIGINAL
# ============================================================

ax1 = plt.subplot(
    1,
    3,
    1
)

ax1.imshow(
    image_resized
)

ax1.set_title(
    f"{SAMPLE_ID} — Original",
    fontsize=13
)

ax1.axis("off")


# ============================================================
# PANEL 2
# WORD-GUIDED ACTIVATION
# ============================================================

ax2 = plt.subplot(
    1,
    3,
    2
)

ax2.imshow(
    image_resized
)


# ------------------------------------------------------------
# Draw top anomalous word regions
# ------------------------------------------------------------

for rank, (_, row) in enumerate(
    top_words.iterrows(),
    start=1
):

    x1 = int(
        row["bbox_x"]
        * scale_x
    )

    y1 = int(
        row["bbox_y"]
        * scale_y
    )

    x2 = int(
        (
            row["bbox_x"]
            + row["bbox_width"]
        )
        * scale_x
    )

    y2 = int(
        (
            row["bbox_y"]
            + row["bbox_height"]
        )
        * scale_y
    )

    # Draw rectangle
    rectangle = plt.Rectangle(
        (
            x1,
            y1
        ),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=1.8
    )

    ax2.add_patch(
        rectangle
    )

    # Label
    label = (
        f"#{rank} "
        f"{row['transcription']}"
    )

    ax2.text(
        x1,
        max(
            8,
            y1 - 3
        ),
        label,
        fontsize=7,
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="none"
        )
    )


ax2.set_title(
    "Top Word-Anomaly Regions",
    fontsize=13
)

ax2.axis("off")


# ============================================================
# PANEL 3
# FINAL REFINED GRAD-CAM
# ============================================================

ax3 = plt.subplot(
    1,
    3,
    3
)

ax3.imshow(
    image_resized
)


# ------------------------------------------------------------
# Display ONLY meaningful activation
# ------------------------------------------------------------

masked_cam = np.ma.masked_where(
    soft_cam <= 0,
    soft_cam
)

ax3.imshow(
    masked_cam,
    cmap="jet",
    alpha=0.65,
    interpolation="bilinear"
)


# ------------------------------------------------------------
# Draw contours around activated regions
# ------------------------------------------------------------

contours, _ = cv2.findContours(
    (
        clean_activation
        * 255
    ).astype(np.uint8),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


for contour in contours:

    area = cv2.contourArea(
        contour
    )

    if area < MIN_COMPONENT_AREA:

        continue

    contour = contour.reshape(
        -1,
        2
    )

    ax3.plot(
        contour[:, 0],
        contour[:, 1],
        linewidth=1.5
    )


ax3.set_title(
    "Refined Anomaly Grad-CAM",
    fontsize=13
)

ax3.axis("off")


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    (
        f"{SAMPLE_ID} — "
        "Anomaly-Specific Explainability"
    ),
    fontsize=16
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94
    ]
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("REFINED GRAD-CAM COMPLETED")
print("=" * 75)

print()

print(
    "Sample:",
    SAMPLE_ID
)

print(
    "Anomaly distance:",
    round(
        float(
            anomaly_distance.item()
        ),
        6
    )
)

print(
    "Top anomalous words:",
    len(top_words)
)

print(
    "Activation threshold percentile:",
    CAM_PERCENTILE
)

print(
    "Meaningful activation regions:",
    len(contours)
)

print()
print(
    "Final visualization:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "Activation mask:"
)

print(
    OUTPUT_MASK
)

print()
print("DONE.")