import os
import cv2
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score, average_precision_score

from model import DysgraphiaModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_CSV = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_form_anomalies.csv"
)

METADATA_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "metadata_all.csv"
)

CLEAN_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "resnet_features.pt"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "best_dysgraphia_encoder.pth"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_evaluation_results.csv"
)

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SYNTHETIC DATASET
# ============================================================

class SyntheticFormDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = (
            PROJECT_ROOT
            / row["synthetic_image"]
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not read: {image_path}"
            )

        # Same preprocessing used by DysgraphiaDataset
        image = cv2.resize(
            image,
            (224, 224)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        image = (
            torch.from_numpy(image)
            .float()
            / 255.0
        )

        # HWC -> CHW
        image = image.permute(
            2,
            0,
            1
        )

        return {
            "image": image,
            "sample_id": str(
                row["sample_id"]
            ),
            "source_form": str(
                row["source_form"]
            ),
            "anomaly_type": str(
                row["anomaly_type"]
            )
        }


# ============================================================
# CACHED FUSION MODEL
# ============================================================

class CachedFusionModel(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.opencv_encoder = (
            model.opencv_encoder
        )

        self.attention = (
            model.attention
        )

        self.projection_head = (
            model.projection_head
        )

        self.opencv_reconstruction = (
            model.opencv_reconstruction
        )

    def forward(
        self,
        image_features,
        opencv_features
    ):

        encoded_opencv = (
            self.opencv_encoder(
                opencv_features
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
            self.attention(
                fused_features
            )
        )

        projection = (
            self.projection_head(
                embedding
            )
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
            "projection":
                projection,
            "reconstructed_opencv":
                reconstructed_opencv
        }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SYNTHETIC FORM ANOMALY EVALUATION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

print("\nLoading synthetic CSV...")

synthetic_df = pd.read_csv(
    SYNTHETIC_CSV
)

metadata_df = pd.read_csv(
    METADATA_CSV
)

print(
    f"Synthetic samples: "
    f"{len(synthetic_df)}"
)

print(
    f"Metadata samples: "
    f"{len(metadata_df)}"
)


# ============================================================
# FIX IAM FORM ID VARIANTS
# ============================================================

FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}


def normalize_form_id(form_id):

    form_id = str(form_id)

    return FORM_ID_MAP.get(
        form_id,
        form_id
    )


synthetic_df["clean_form_id"] = (
    synthetic_df["source_form"]
    .apply(normalize_form_id)
)


# Check that every synthetic source has metadata
metadata_ids = set(
    metadata_df["form_id"]
    .astype(str)
)

missing = synthetic_df[
    ~synthetic_df["clean_form_id"].isin(
        metadata_ids
    )
]

if len(missing) > 0:

    print("\nERROR: Missing metadata for:")

    print(
        missing[
            "source_form"
        ].unique()
    )

    raise RuntimeError(
        "Some synthetic forms cannot "
        "be matched to IAM metadata."
    )

print(
    "\nAll 1000 synthetic forms "
    "matched to clean IAM metadata."
)


# ============================================================
# LOAD CLEAN CACHED FEATURES
# ============================================================

print("\nLoading clean ResNet features...")

clean_data = torch.load(
    CLEAN_FEATURE_FILE,
    map_location="cpu"
)

clean_image_features = (
    clean_data["image_features"]
)

clean_opencv_features = (
    clean_data["opencv_features"]
)

clean_sample_ids = [
    str(x)
    for x in clean_data["sample_ids"]
]

print(
    "Clean image features:",
    clean_image_features.shape
)

print(
    "Clean OpenCV features:",
    clean_opencv_features.shape
)


# ============================================================
# CREATE CLEAN FEATURE LOOKUP
# ============================================================

clean_feature_lookup = {}

for i, sample_id in enumerate(
    clean_sample_ids
):

    clean_feature_lookup[
        sample_id
    ] = clean_opencv_features[i]


# Match metadata form_id -> sample_id
metadata_sample_lookup = {}

for _, row in metadata_df.iterrows():

    form_id = str(
        row["form_id"]
    )

    sample_id = str(
        row["sample_id"]
    )

    metadata_sample_lookup[
        form_id
    ] = sample_id


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading trained checkpoint...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

feature_mean = (
    checkpoint["feature_mean"]
    .to(DEVICE)
)

feature_std = (
    checkpoint["feature_std"]
    .to(DEVICE)
)

print(
    "\nCheckpoint feature names:"
)

print(
    checkpoint["feature_names"]
)

print(
    "\nFeature normalization loaded."
)


# ============================================================
# LOAD TRAINED FUSION MODEL
# ============================================================

base_model = DysgraphiaModel(
    pretrained=False,
    freeze_backbone=True
)

base_model = base_model.to(
    DEVICE
)

fusion_model = CachedFusionModel(
    base_model
)

fusion_model = fusion_model.to(
    DEVICE
)

fusion_model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

fusion_model.eval()

print(
    "Trained fusion model loaded."
)


# ============================================================
# CREATE CLEAN REFERENCE EMBEDDING
# ============================================================

print(
    "\nCreating clean reference embedding..."
)

clean_embeddings = []

with torch.no_grad():

    for start in range(
        0,
        len(clean_image_features),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(clean_image_features)
        )

        image_features = (
            clean_image_features[
                start:end
            ].to(DEVICE)
        )

        opencv_features = (
            clean_opencv_features[
                start:end
            ].to(DEVICE)
        )

        # EXACT training normalization
        opencv_features = (
            opencv_features
            - feature_mean
        ) / feature_std

        output = fusion_model(
            image_features,
            opencv_features
        )

        clean_embeddings.append(
            output["embedding"]
            .cpu()
        )


clean_embeddings = torch.cat(
    clean_embeddings,
    dim=0
)

reference_embedding = (
    clean_embeddings.mean(
        dim=0
    )
)

print(
    "Reference embedding:",
    reference_embedding.shape
)


# ============================================================
# LOAD PRETRAINED RESNET50
# ============================================================

print(
    "\nLoading pretrained ResNet50..."
)

weights = ResNet50_Weights.DEFAULT

resnet = resnet50(
    weights=weights
)

resnet.fc = torch.nn.Identity()

resnet = resnet.to(
    DEVICE
)

resnet.eval()

for parameter in resnet.parameters():

    parameter.requires_grad = False


# EXACT ImageNet normalization
mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE
).view(
    1, 3, 1, 1
)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE
).view(
    1, 3, 1, 1
)


# ============================================================
# SYNTHETIC DATA LOADER
# ============================================================

dataset = SyntheticFormDataset(
    synthetic_df
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# EXTRACT SYNTHETIC RESNET FEATURES
# ============================================================

print(
    "\nExtracting synthetic ResNet features..."
)

synthetic_image_features = []

with torch.no_grad():

    for batch_index, batch in enumerate(
        loader
    ):

        images = batch[
            "image"
        ].to(DEVICE)

        # Same ImageNet normalization
        images = (
            images - mean
        ) / std

        features = resnet(
            images
        )

        synthetic_image_features.append(
            features.cpu()
        )

        print(
            f"Batch "
            f"{batch_index + 1}/"
            f"{len(loader)}"
        )


synthetic_image_features = (
    torch.cat(
        synthetic_image_features,
        dim=0
    )
)

print(
    "\nSynthetic ResNet features:",
    synthetic_image_features.shape
)


# ============================================================
# GET CORRESPONDING CLEAN OPENCV FEATURES
# ============================================================

print(
    "\nMatching OpenCV features..."
)

synthetic_opencv_features = []

for _, row in synthetic_df.iterrows():

    form_id = str(
        row["clean_form_id"]
    )

    sample_id = (
        metadata_sample_lookup[
            form_id
        ]
    )

    if sample_id not in clean_feature_lookup:

        raise RuntimeError(
            f"Cached feature missing "
            f"for {sample_id}"
        )

    synthetic_opencv_features.append(
        clean_feature_lookup[
            sample_id
        ]
    )


synthetic_opencv_features = torch.stack(
    synthetic_opencv_features
)

print(
    "Synthetic OpenCV features:",
    synthetic_opencv_features.shape
)


# ============================================================
# GENERATE SYNTHETIC EMBEDDINGS
# ============================================================

print(
    "\nGenerating synthetic embeddings..."
)

synthetic_embeddings = []

with torch.no_grad():

    for start in range(
        0,
        len(synthetic_image_features),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(synthetic_image_features)
        )

        image_features = (
            synthetic_image_features[
                start:end
            ].to(DEVICE)
        )

        opencv_features = (
            synthetic_opencv_features[
                start:end
            ].to(DEVICE)
        )

        # EXACT training normalization
        opencv_features = (
            opencv_features
            - feature_mean
        ) / feature_std

        output = fusion_model(
            image_features,
            opencv_features
        )

        synthetic_embeddings.append(
            output["embedding"]
            .cpu()
        )


synthetic_embeddings = torch.cat(
    synthetic_embeddings,
    dim=0
)


# ============================================================
# ANOMALY SCORES
# ============================================================

print(
    "\nCalculating anomaly scores..."
)

reference_embedding = (
    reference_embedding.cpu()
)

raw_scores = torch.norm(
    synthetic_embeddings
    - reference_embedding,
    dim=1
).numpy()


# ============================================================
# CREATE RESULTS
# ============================================================

results_df = synthetic_df[
    [
        "sample_id",
        "source_form",
        "synthetic_image",
        "ground_truth_mask",
        "anomaly_type",
        "label",
        "mask_pixels"
    ]
].copy()

results_df[
    "raw_anomaly_score"
] = raw_scores


# 0-100 normalized score
score_min = raw_scores.min()
score_max = raw_scores.max()

if score_max > score_min:

    results_df[
        "anomaly_score_0_100"
    ] = (
        (raw_scores - score_min)
        /
        (score_max - score_min)
        * 100
    )

else:

    results_df[
        "anomaly_score_0_100"
    ] = 0.0


# ============================================================
# ROC-AUC / AP
# ============================================================

# All synthetic samples are anomalous.
# Therefore, ROC-AUC/AP cannot be calculated using
# ONLY these 1000 positive samples.
#
# We need clean samples + synthetic anomalies
# as the binary evaluation set.

print(
    "\nPreparing clean-vs-anomalous evaluation..."
)


# Calculate clean scores against same reference
clean_raw_scores = torch.norm(
    clean_embeddings
    - reference_embedding,
    dim=1
).numpy()


# Synthetic scores are positive class
synthetic_scores = raw_scores


all_scores = np.concatenate(
    [
        clean_raw_scores,
        synthetic_scores
    ]
)

all_labels = np.concatenate(
    [
        np.zeros(
            len(clean_raw_scores)
        ),
        np.ones(
            len(synthetic_scores)
        )
    ]
)


roc_auc = roc_auc_score(
    all_labels,
    all_scores
)

average_precision = (
    average_precision_score(
        all_labels,
        all_scores
    )
)


print(
    "\n" + "=" * 70
)

print(
    "QUANTITATIVE RESULTS"
)

print(
    "=" * 70
)

print(
    f"\nClean samples: "
    f"{len(clean_raw_scores)}"
)

print(
    f"Synthetic anomalies: "
    f"{len(synthetic_scores)}"
)

print(
    f"\nROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"Average Precision: "
    f"{average_precision:.4f}"
)


# ============================================================
# RESULTS BY ANOMALY TYPE
# ============================================================

print(
    "\nMean anomaly score by type:"
)

type_summary = (
    results_df
    .groupby(
        "anomaly_type"
    )[
        "raw_anomaly_score"
    ]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "max"
        ]
    )
)

print(
    type_summary
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_FILE.parent,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nEvaluation complete."
)