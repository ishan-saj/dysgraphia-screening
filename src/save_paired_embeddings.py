from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = PROJECT_ROOT / "data" / "metadata" / "resnet_features.pt"
CHECKPOINT_FILE = PROJECT_ROOT / "best_dysgraphia_encoder.pth"
SYNTHETIC_CSV = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_form_anomalies.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "paired_embeddings.npz"
)

DEVICE = torch.device("cpu")


# ============================================================
# FORM ID MAP
# ============================================================

FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}


# ============================================================
# MODEL
# ============================================================

class OpenCVEncoder(nn.Module):

    def __init__(self, input_dim=9, output_dim=64):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.network(x)


class FeatureAttention(nn.Module):

    def __init__(self, input_dim=2112, embedding_dim=256):

        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):

        attention_scores = self.attention(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        attended_features = (
            x * attention_weights
        )

        embedding = self.projection(
            attended_features
        )

        return embedding


class CachedFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.opencv_encoder = OpenCVEncoder()

        self.attention = FeatureAttention()

    def forward(self, image_features, opencv_features):

        encoded_opencv = self.opencv_encoder(
            opencv_features
        )

        fused = torch.cat(
            [
                image_features,
                encoded_opencv
            ],
            dim=1
        )

        embedding = self.attention(
            fused
        )

        return embedding


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image: {path}"
        )

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )

    image = torch.from_numpy(
        image
    ).float() / 255.0

    image = image.permute(
        2,
        0,
        1
    )

    return image


# ============================================================
# RESNET FEATURE
# ============================================================

def get_resnet_feature(model, path):

    image = load_image(
        path
    ).unsqueeze(0)

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    image = (
        image - mean
    ) / std

    with torch.no_grad():

        feature = model(image)

    return feature.squeeze(0)


# ============================================================
# LOAD CACHED FEATURES
# ============================================================

print("Loading cached features...")

cached = torch.load(
    FEATURE_FILE,
    map_location=DEVICE,
    weights_only=False
)

clean_image_features = (
    cached["image_features"]
    .float()
)

clean_opencv_features = (
    cached["opencv_features"]
    .float()
)

sample_ids = [
    str(x)
    for x in cached["sample_ids"]
]

print(
    "Clean samples:",
    len(sample_ids)
)


# ============================================================
# CREATE CLEAN FORM → INDEX MAP
# ============================================================
#
# NEW:
# This lets us record exactly which clean embedding
# each synthetic sample came from.
#
# Example:
#
# source_form = "a05-094"
# source_index = 123
#
# means synthetic sample was generated from
# clean_embeddings[123].
#
# ============================================================

clean_index_by_id = {
    sample_ids[i]: i
    for i in range(len(sample_ids))
}


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=False
)

feature_mean = checkpoint[
    "feature_mean"
].float()

feature_std = checkpoint[
    "feature_std"
].float()


model = CachedFusionModel()

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=False
)

model.eval()

print("Fusion model loaded.")


# ============================================================
# CLEAN EMBEDDINGS
# ============================================================

print("\nCalculating clean embeddings...")

normalized_opencv = (
    clean_opencv_features
    - feature_mean
) / (
    feature_std + 1e-8
)

with torch.no_grad():

    clean_embeddings = model(
        clean_image_features,
        normalized_opencv
    )

clean_embeddings = (
    clean_embeddings
    .cpu()
    .numpy()
)

print(
    "Clean embeddings:",
    clean_embeddings.shape
)


# ============================================================
# MAP CLEAN FEATURES
# ============================================================

opencv_by_id = {
    sample_ids[i]: clean_opencv_features[i]
    for i in range(len(sample_ids))
}


# ============================================================
# LOAD RESNET
# ============================================================

print("\nLoading ResNet50...")

resnet = resnet50(
    weights=ResNet50_Weights.DEFAULT
)

resnet.fc = nn.Identity()

resnet.eval()

print("ResNet50 loaded.")


# ============================================================
# LOAD SYNTHETIC DATA
# ============================================================

synthetic = pd.read_csv(
    SYNTHETIC_CSV
)

print(
    "\nSynthetic samples:",
    len(synthetic)
)


synthetic_embeddings = []

synthetic_types = []

source_forms = []

# ============================================================
# NEW:
# Exact index of the clean embedding corresponding
# to each synthetic sample.
# ============================================================

source_indices = []


# ============================================================
# PROCESS SYNTHETIC IMAGES
# ============================================================

for index, row in synthetic.iterrows():

    source_form = str(
        row["source_form"]
    )

    source_form = FORM_ID_MAP.get(
        source_form,
        source_form
    )

    if source_form not in opencv_by_id:

        print(
            "Skipping unmatched form:",
            source_form
        )

        continue

    # --------------------------------------------------------
    # NEW:
    # Find exact clean embedding index
    # --------------------------------------------------------

    source_index = clean_index_by_id[source_form]

    image_path = (
        PROJECT_ROOT
        / row["synthetic_image"]
    )

    if not image_path.exists():

        image_path = (
            PROJECT_ROOT
            / "data"
            / "synthetic_form_anomalies"
            / "images"
            / Path(
                row["synthetic_image"]
            ).name
        )

    # ResNet feature
    image_feature = get_resnet_feature(
        resnet,
        image_path
    )

    image_feature = (
        image_feature
        .unsqueeze(0)
    )

    # Original clean OpenCV features
    opencv_feature = (
        opencv_by_id[source_form]
        .unsqueeze(0)
    )

    opencv_feature = (
        opencv_feature
        - feature_mean
    ) / (
        feature_std + 1e-8
    )

    # Fusion embedding
    with torch.no_grad():

        embedding = model(
            image_feature,
            opencv_feature
        )

    embedding = (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
    )

    synthetic_embeddings.append(
        embedding
    )

    synthetic_types.append(
        row["anomaly_type"]
    )

    source_forms.append(
        source_form
    )

    # --------------------------------------------------------
    # NEW:
    # Save the exact clean embedding index.
    # --------------------------------------------------------

    source_indices.append(
        source_index
    )

    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/{len(synthetic)}"
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

synthetic_embeddings = np.array(
    synthetic_embeddings
)

synthetic_types = np.array(
    synthetic_types
)

source_forms = np.array(
    source_forms
)

source_indices = np.array(
    source_indices,
    dtype=np.int64
)


# ============================================================
# SAVE
# ============================================================

np.savez(
    OUTPUT_FILE,

    clean_embeddings=clean_embeddings,

    synthetic_embeddings=synthetic_embeddings,

    synthetic_types=synthetic_types,

    source_forms=source_forms,

    # NEW
    source_indices=source_indices
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n")
print("=" * 60)
print("EMBEDDINGS SAVED")
print("=" * 60)

print(
    "Clean:",
    clean_embeddings.shape
)

print(
    "Synthetic:",
    synthetic_embeddings.shape
)

print(
    "Anomaly types:",
    len(synthetic_types)
)

print(
    "Source indices:",
    source_indices.shape
)

print(
    "Unique source forms:",
    len(np.unique(source_forms))
)

print("\nSaved to:")

print(
    OUTPUT_FILE
)