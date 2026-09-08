import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model import DysgraphiaModel


# ============================================================
# Configuration
# ============================================================

FEATURE_FILE = "../data/metadata/resnet_features.pt"

MODEL_FILE = "../best_dysgraphia_encoder.pth"

OUTPUT_FILE = "../data/metadata/anomaly_scores.csv"

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Cached Feature Dataset
# ============================================================

class CachedFeatureDataset(Dataset):

    def __init__(self, feature_file):

        data = torch.load(
            feature_file,
            map_location="cpu"
        )

        self.image_features = (
            data["image_features"].float()
        )

        self.opencv_features = (
            data["opencv_features"].float()
        )

        self.sample_ids = data["sample_ids"]

        self.image_paths = data["image_paths"]

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, index):

        return {
            "image_features":
                self.image_features[index],

            "opencv_features":
                self.opencv_features[index],

            "sample_id":
                self.sample_ids[index],

            "image_path":
                self.image_paths[index],
        }


# ============================================================
# Cached Fusion Model
# ============================================================

class CachedFusionModel(nn.Module):

    def __init__(self, original_model):

        super().__init__()

        self.opencv_encoder = (
            original_model.opencv_encoder
        )

        self.attention = (
            original_model.attention
        )

        self.projection_head = (
            original_model.projection_head
        )

        self.opencv_reconstruction = (
            original_model.opencv_reconstruction
        )

    def forward(
        self,
        image_features,
        opencv_features
    ):

        encoded_opencv = self.opencv_encoder(
            opencv_features
        )

        fused_features = torch.cat(
            [
                image_features,
                encoded_opencv
            ],
            dim=1
        )

        embedding, attention_weights = (
            self.attention(fused_features)
        )

        projection = self.projection_head(
            embedding
        )

        reconstructed_opencv = (
            self.opencv_reconstruction(
                embedding
            )
        )

        return {
            "embedding": embedding,

            "projection": projection,

            "attention_weights":
                attention_weights,

            "reconstructed_opencv":
                reconstructed_opencv,
        }


# ============================================================
# Load Dataset
# ============================================================

print(f"Using device: {DEVICE}")

dataset = CachedFeatureDataset(
    FEATURE_FILE
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"Loaded {len(dataset)} samples."
)


# ============================================================
# Load Model
# ============================================================

print("\nLoading trained model...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

original_model = DysgraphiaModel(
    pretrained=False,
    freeze_backbone=True
)

model = CachedFusionModel(
    original_model
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Load Normalization Parameters
# ============================================================

feature_mean = checkpoint[
    "feature_mean"
].float()

feature_std = checkpoint[
    "feature_std"
].float()

feature_std[
    feature_std < 1e-8
] = 1.0


# ============================================================
# Generate Embeddings
# ============================================================

all_embeddings = []

all_sample_ids = []

all_image_paths = []

print("\nGenerating embeddings...")
print("=" * 50)


with torch.no_grad():

    for batch_idx, batch in enumerate(
        dataloader
    ):

        image_features = batch[
            "image_features"
        ].to(DEVICE)

        opencv_features = batch[
            "opencv_features"
        ]

        # Apply same normalization
        # used during training

        opencv_features = (
            opencv_features
            - feature_mean
        ) / feature_std

        opencv_features = (
            opencv_features.to(DEVICE)
        )

        output = model(
            image_features,
            opencv_features
        )

        embeddings = output[
            "embedding"
        ]

        all_embeddings.append(
            embeddings.cpu()
        )

        all_sample_ids.extend(
            batch["sample_id"]
        )

        all_image_paths.extend(
            batch["image_path"]
        )

        print(
            f"Batch {batch_idx + 1}/"
            f"{len(dataloader)}"
        )


# ============================================================
# Combine Embeddings
# ============================================================

embeddings = torch.cat(
    all_embeddings,
    dim=0
)

print("\nEmbedding shape:")
print(embeddings.shape)


# ============================================================
# Calculate Reference Distribution
# ============================================================

print("\nCalculating reference distribution...")

reference_embedding = embeddings.mean(
    dim=0
)

print(
    "Reference embedding shape:",
    reference_embedding.shape
)


# ============================================================
# Calculate Anomaly Scores
# ============================================================

print("\nCalculating anomaly scores...")

# Euclidean distance from the
# reference embedding

distances = torch.norm(
    embeddings - reference_embedding,
    dim=1
)

anomaly_scores = distances.numpy()


# ============================================================
# Normalize Scores to 0-100
# ============================================================

min_score = anomaly_scores.min()

max_score = anomaly_scores.max()

if max_score > min_score:

    normalized_scores = (
        (anomaly_scores - min_score)
        /
        (max_score - min_score)
    ) * 100

else:

    normalized_scores = np.zeros(
        len(anomaly_scores)
    )


# ============================================================
# Create Results DataFrame
# ============================================================

results = pd.DataFrame({

    "sample_id":
        all_sample_ids,

    "image_path":
        all_image_paths,

    "anomaly_score":
        anomaly_scores,

    "anomaly_score_0_100":
        normalized_scores,
})


# ============================================================
# Sort by Anomaly Score
# ============================================================

results = results.sort_values(
    by="anomaly_score",
    ascending=False
)


# ============================================================
# Save Results
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display Results
# ============================================================

print("\n")
print("=" * 60)
print("ANOMALY DETECTION RESULTS")
print("=" * 60)

print("\nMost unusual samples:")

print(
    results[
        [
            "sample_id",
            "anomaly_score",
            "anomaly_score_0_100"
        ]
    ].head(10).to_string(
        index=False
    )
)


print("\nMost typical samples:")

print(
    results[
        [
            "sample_id",
            "anomaly_score",
            "anomaly_score_0_100"
        ]
    ].tail(10).to_string(
        index=False
    )
)


print("\n")
print("=" * 60)

print(
    f"Results saved to:\n{OUTPUT_FILE}"
)

print(
    f"\nTotal samples evaluated: "
    f"{len(results)}"
)

print(
    f"Minimum anomaly score: "
    f"{anomaly_scores.min():.4f}"
)

print(
    f"Maximum anomaly score: "
    f"{anomaly_scores.max():.4f}"
)

print(
    f"Mean anomaly score: "
    f"{anomaly_scores.mean():.4f}"
)

print("=" * 60)