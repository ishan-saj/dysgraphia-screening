import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np

from model import DysgraphiaModel


# ============================================================
# Configuration
# ============================================================

FEATURE_FILE = "../data/metadata/resnet_features.pt"
CHECKPOINT_FILE = "../best_dysgraphia_encoder.pth"
OUTPUT_FILE = "../data/metadata/anomaly_scores.csv"

VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

N_NEIGHBORS = 20

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

        self.image_features = data["image_features"].float()
        self.opencv_features = data["opencv_features"].float()

        self.sample_ids = data["sample_ids"]
        self.image_paths = data["image_paths"]

        print(
            f"Loaded cached features: "
            f"{len(self.image_features)} samples"
        )

        print(
            f"Image features: "
            f"{self.image_features.shape}"
        )

        print(
            f"OpenCV features: "
            f"{self.opencv_features.shape}"
        )

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
            "attention_weights":
                attention_weights,
            "projection": projection,
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

print(
    f"\nTotal samples: {len(dataset)}"
)


# ============================================================
# Reproduce EXACT training split
# ============================================================

validation_size = int(
    len(dataset) * VALIDATION_SPLIT
)

training_size = (
    len(dataset) - validation_size
)

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = random_split(
    dataset,
    [
        training_size,
        validation_size
    ],
    generator=generator
)

train_indices = train_dataset.indices
validation_indices = validation_dataset.indices

print("\nDataset split:")
print("-" * 50)

print(
    f"Training samples: "
    f"{len(train_indices)}"
)

print(
    f"Validation samples: "
    f"{len(validation_indices)}"
)


# ============================================================
# Load trained model checkpoint
# ============================================================

print("\nLoading trained model...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE
)


# ============================================================
# Recreate model architecture
# ============================================================

original_model = DysgraphiaModel(
    pretrained=False,
    freeze_backbone=True
)

model = CachedFusionModel(
    original_model
).to(DEVICE)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Load TRAINING normalization statistics
# ============================================================

feature_mean = checkpoint[
    "feature_mean"
].float()

feature_std = checkpoint[
    "feature_std"
].float()

print(
    "\nLoaded OpenCV normalization "
    "statistics from training."
)


# ============================================================
# Normalize OpenCV features
# ============================================================

normalized_opencv = (
    dataset.opencv_features
    - feature_mean
) / feature_std


# ============================================================
# Generate 256-D embeddings
# ============================================================

print("\nGenerating 256-D embeddings...")

all_embeddings = []

with torch.no_grad():

    for index in range(len(dataset)):

        image_features = (
            dataset.image_features[index]
            .unsqueeze(0)
            .to(DEVICE)
        )

        opencv_features = (
            normalized_opencv[index]
            .unsqueeze(0)
            .to(DEVICE)
        )

        output = model(
            image_features,
            opencv_features
        )

        embedding = (
            output["embedding"]
            .squeeze(0)
            .cpu()
            .numpy()
        )

        all_embeddings.append(
            embedding
        )


embedding_array = np.asarray(
    all_embeddings
)

print(
    f"Embedding matrix: "
    f"{embedding_array.shape}"
)


# ============================================================
# Separate TRAIN and VALIDATION embeddings
# ============================================================

train_embeddings = (
    embedding_array[train_indices]
)

validation_embeddings = (
    embedding_array[validation_indices]
)

print(
    f"Training embeddings: "
    f"{train_embeddings.shape}"
)

print(
    f"Validation embeddings: "
    f"{validation_embeddings.shape}"
)


# ============================================================
# LOF MODEL
# ============================================================

n_neighbors = min(
    N_NEIGHBORS,
    len(train_embeddings) - 1
)

print("\nFitting LOF...")
print(
    f"LOF neighbours: {n_neighbors}"
)

lof = LocalOutlierFactor(
    n_neighbors=n_neighbors,
    contamination="auto",
    metric="euclidean",
    novelty=True
)


# ============================================================
# FIT ONLY ON TRAINING EMBEDDINGS
# ============================================================

lof.fit(
    train_embeddings
)

print(
    "LOF fitted using training embeddings only."
)


# ============================================================
# Score VALIDATION samples
# ============================================================

validation_normality_scores = (
    lof.score_samples(
        validation_embeddings
    )
)

# sklearn's score_samples:
# higher = more normal
#
# Reverse sign:
# higher = more anomalous

validation_anomaly_scores = (
    -validation_normality_scores
)


# ============================================================
# Normalize validation scores to 0-100
# ============================================================

score_min = (
    validation_anomaly_scores.min()
)

score_max = (
    validation_anomaly_scores.max()
)

if score_max > score_min:

    validation_scores_0_100 = (
        (
            validation_anomaly_scores
            - score_min
        )
        /
        (
            score_max
            - score_min
        )
        * 100
    )

else:

    validation_scores_0_100 = np.zeros(
        len(validation_anomaly_scores)
    )


# ============================================================
# Create results
# ============================================================

results = []

for position, dataset_index in enumerate(
    validation_indices
):

    results.append({

        "sample_id":
            dataset.sample_ids[
                dataset_index
            ],

        "image_path":
            dataset.image_paths[
                dataset_index
            ],

        "dataset_split":
            "validation",

        "anomaly_score":
            float(
                validation_anomaly_scores[
                    position
                ]
            ),

        "anomaly_score_0_100":
            float(
                validation_scores_0_100[
                    position
                ]
            ),

        "lof_score_samples":
            float(
                validation_normality_scores[
                    position
                ]
            ),

        "lof_n_neighbors":
            n_neighbors
    })


# ============================================================
# Sort by anomaly score
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "anomaly_score",
    ascending=False
).reset_index(drop=True)


# ============================================================
# Save
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Display results
# ============================================================

print("\n")
print("=" * 60)
print("LOF VALIDATION RESULTS")
print("=" * 60)

print(
    f"Samples scored: "
    f"{len(results_df)}"
)

print(
    f"Mean anomaly score: "
    f"{results_df['anomaly_score'].mean():.6f}"
)

print(
    f"Minimum anomaly score: "
    f"{results_df['anomaly_score'].min():.6f}"
)

print(
    f"Maximum anomaly score: "
    f"{results_df['anomaly_score'].max():.6f}"
)

print("\nTop 10 validation samples:")

print(
    results_df[
        [
            "sample_id",
            "anomaly_score",
            "anomaly_score_0_100"
        ]
    ].head(10).to_string(
        index=False
    )
)

print("\n")
print(
    f"Results saved to:\n"
    f"{OUTPUT_FILE}"
)

print("\nEvaluation completed.")