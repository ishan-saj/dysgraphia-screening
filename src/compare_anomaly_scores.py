from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = PROJECT_ROOT / "data" / "metadata" / "resnet_features.pt"
CHECKPOINT_FILE = PROJECT_ROOT / "best_dysgraphia_encoder.pth"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"

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
    / "anomaly_score_comparison.csv"
)

DEVICE = torch.device("cpu")


# ============================================================
# FORM ID FIX
# ============================================================

FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}


# ============================================================
# CACHED FUSION MODEL
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

        attended_features = x * attention_weights

        embedding = self.projection(
            attended_features
        )

        return embedding, attention_weights


class CachedFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.opencv_encoder = OpenCVEncoder(
            input_dim=9,
            output_dim=64
        )

        self.attention = FeatureAttention(
            input_dim=2112,
            embedding_dim=256
        )

        self.projection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.opencv_reconstruction = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, image_features, opencv_features):

        encoded_opencv = self.opencv_encoder(
            opencv_features
        )

        fused = torch.cat(
            [image_features, encoded_opencv],
            dim=1
        )

        embedding, attention = self.attention(
            fused
        )

        projection = self.projection_head(
            embedding
        )

        reconstruction = self.opencv_reconstruction(
            embedding
        )

        return {
            "embedding": embedding,
            "projection": projection,
            "attention": attention,
            "reconstruction": reconstruction
        }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
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
        2, 0, 1
    )

    return image


# ============================================================
# RESNET FEATURE EXTRACTION
# ============================================================

def extract_resnet_feature(model, image_path):

    image = load_image(
        image_path
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

        feature = model(
            image
        )

    return feature.squeeze(0)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cached features...")

cached = torch.load(
    FEATURE_FILE,
    map_location=DEVICE,
    weights_only=False
)

cached_image_features = cached[
    "image_features"
].float()

cached_opencv_features = cached[
    "opencv_features"
].float()

sample_ids = cached[
    "sample_ids"
]

sample_ids = [
    str(x)
    for x in sample_ids
]

print(
    "Cached image features:",
    cached_image_features.shape
)


print("\nLoading metadata...")

metadata = pd.read_csv(
    METADATA_FILE
)

metadata["sample_id"] = (
    metadata["sample_id"]
    .astype(str)
)

metadata_index = {
    row["sample_id"]: row
    for _, row in metadata.iterrows()
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


fusion_model = CachedFusionModel()

fusion_model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

fusion_model.to(DEVICE)
fusion_model.eval()

print("Fusion model loaded.")


# ============================================================
# CLEAN EMBEDDINGS
# ============================================================

print("\nCalculating clean embeddings...")

normalized_clean_opencv = (
    cached_opencv_features
    - feature_mean
) / (
    feature_std + 1e-8
)

with torch.no_grad():

    clean_output = fusion_model(
        cached_image_features,
        normalized_clean_opencv
    )

clean_embeddings = (
    clean_output["embedding"]
    .cpu()
    .numpy()
)

print(
    "Clean embeddings:",
    clean_embeddings.shape
)


# ============================================================
# REFERENCE EMBEDDING
# ============================================================

reference_embedding = (
    clean_embeddings.mean(axis=0)
)


# ============================================================
# MAHALANOBIS MODEL
# ============================================================

print("\nBuilding Mahalanobis model...")

covariance = np.cov(
    clean_embeddings,
    rowvar=False
)

# Regularization for numerical stability
regularization = (
    np.eye(covariance.shape[0])
    * 1e-3
)

covariance += regularization

covariance_inverse = np.linalg.pinv(
    covariance
)


# ============================================================
# KNN MODEL
# ============================================================

print("Building k-NN model...")

knn = NearestNeighbors(
    n_neighbors=6,
    metric="euclidean"
)

knn.fit(
    clean_embeddings
)


# ============================================================
# RESNET
# ============================================================

print("\nLoading pretrained ResNet50...")

resnet = resnet50(
    weights=ResNet50_Weights.DEFAULT
)

resnet.fc = nn.Identity()

resnet.to(DEVICE)
resnet.eval()

print("ResNet50 loaded.")


# ============================================================
# SYNTHETIC DATA
# ============================================================

synthetic = pd.read_csv(
    SYNTHETIC_CSV
)

print(
    "\nSynthetic anomalies:",
    len(synthetic)
)


# ============================================================
# MAP CLEAN FORM
# ============================================================

clean_embedding_by_id = {
    sample_ids[i]: clean_embeddings[i]
    for i in range(len(sample_ids))
}

clean_opencv_by_id = {
    sample_ids[i]: cached_opencv_features[i]
    for i in range(len(sample_ids))
}


# ============================================================
# SCORE FUNCTIONS
# ============================================================

def euclidean_score(x):

    return np.linalg.norm(
        x - reference_embedding
    )


def cosine_score(x):

    numerator = np.dot(
        x,
        reference_embedding
    )

    denominator = (
        np.linalg.norm(x)
        * np.linalg.norm(reference_embedding)
        + 1e-8
    )

    similarity = (
        numerator / denominator
    )

    return 1.0 - similarity


def mahalanobis_score(x):

    difference = (
        x - reference_embedding
    )

    return np.sqrt(
        difference
        @ covariance_inverse
        @ difference.T
    )


# ============================================================
# PROCESS SYNTHETIC DATA
# ============================================================

results = []

for index, row in synthetic.iterrows():

    source_form = str(
        row["source_form"]
    )

    source_form = FORM_ID_MAP.get(
        source_form,
        source_form
    )

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

    if source_form not in clean_opencv_by_id:

        continue

    # --------------------------------------------------------
    # ResNet feature
    # --------------------------------------------------------

    resnet_feature = extract_resnet_feature(
        resnet,
        image_path
    )

    resnet_feature = (
        resnet_feature
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Original clean OpenCV features
    # --------------------------------------------------------

    opencv_feature = (
        clean_opencv_by_id[
            source_form
        ]
        .unsqueeze(0)
    )

    normalized_opencv = (
        opencv_feature
        - feature_mean
    ) / (
        feature_std + 1e-8
    )

    # --------------------------------------------------------
    # Synthetic embedding
    # --------------------------------------------------------

    with torch.no_grad():

        output = fusion_model(
            resnet_feature,
            normalized_opencv
        )

    synthetic_embedding = (
        output["embedding"]
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    euclidean = euclidean_score(
        synthetic_embedding
    )

    cosine = cosine_score(
        synthetic_embedding
    )

    mahalanobis = mahalanobis_score(
        synthetic_embedding
    )

    # kNN
    distances, _ = knn.kneighbors(
        synthetic_embedding.reshape(1, -1)
    )

    knn_score = distances[0, -1]

    results.append({

        "source_form": source_form,

        "anomaly_type": row[
            "anomaly_type"
        ],

        "label": 1,

        "euclidean_score": euclidean,

        "cosine_score": cosine,

        "mahalanobis_score": mahalanobis,

        "knn_score": knn_score
    })

    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/{len(synthetic)}"
        )


# ============================================================
# CLEAN SCORES
# ============================================================

print("\nCalculating clean scores...")

for embedding in clean_embeddings:

    euclidean = euclidean_score(
        embedding
    )

    cosine = cosine_score(
        embedding
    )

    mahalanobis = mahalanobis_score(
        embedding
    )

    distances, _ = knn.kneighbors(
        embedding.reshape(1, -1)
    )

    knn_score = distances[0, -1]

    results.append({

        "source_form": "clean",

        "anomaly_type": "clean",

        "label": 0,

        "euclidean_score": euclidean,

        "cosine_score": cosine,

        "mahalanobis_score": mahalanobis,

        "knn_score": knn_score
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# METRICS
# ============================================================

y_true = results_df[
    "label"
].values

methods = [
    "euclidean_score",
    "cosine_score",
    "mahalanobis_score",
    "knn_score"
]

print("\n")
print("=" * 60)
print("ANOMALY SCORE COMPARISON")
print("=" * 60)

for method in methods:

    scores = results_df[
        method
    ].values

    auc = roc_auc_score(
        y_true,
        scores
    )

    ap = average_precision_score(
        y_true,
        scores
    )

    print(
        f"{method:22s} "
        f"ROC-AUC = {auc:.4f}   "
        f"AP = {ap:.4f}"
    )


# ============================================================
# TYPE-WISE RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("MEAN SCORE BY ANOMALY TYPE")
print("=" * 60)

synthetic_results = results_df[
    results_df["label"] == 1
]

print(
    synthetic_results.groupby(
        "anomaly_type"
    )[methods]
    .mean()
)


print("\nResults saved to:")

print(
    OUTPUT_FILE
)