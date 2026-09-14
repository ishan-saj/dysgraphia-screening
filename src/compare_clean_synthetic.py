from pathlib import Path
import pandas as pd
import torch
from torchvision.models import resnet50, ResNet50_Weights
import cv2


from model import DysgraphiaModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"
)

SYNTHETIC_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_form_anomalies.csv"
)

FEATURE_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "resnet_features.pt"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT / "best_dysgraphia_encoder.pth"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "paired_clean_vs_synthetic.csv"
)

SYNTHETIC_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "images"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("Device:", device)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)
synthetic = pd.read_csv(SYNTHETIC_FILE)

print("Clean metadata:", len(metadata))
print("Synthetic samples:", len(synthetic))


# ============================================================
# SOURCE FORM ID FIX
# ============================================================

FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}

synthetic["clean_form_id"] = (
    synthetic["source_form"].replace(FORM_ID_MAP)
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device,
    weights_only=False
)

feature_mean = checkpoint["feature_mean"].float()
feature_std = checkpoint["feature_std"].float()

print("Checkpoint loaded")


# ============================================================
# CREATE MODEL TEMPLATE
# ============================================================

model_template = DysgraphiaModel(
    pretrained=False,
    freeze_backbone=True
)


# ============================================================
# CACHED FEATURE MODEL
# ============================================================

class CachedFusionModel(torch.nn.Module):

    def __init__(self, template):
        super().__init__()

        self.opencv_encoder = template.opencv_encoder
        self.attention = template.attention
        self.projection_head = template.projection_head
        self.opencv_reconstruction = (
            template.opencv_reconstruction
        )

    def forward(self, image_features, opencv_features):

        encoded_opencv = self.opencv_encoder(
            opencv_features
        )

        fused_features = torch.cat(
            [image_features, encoded_opencv],
            dim=1
        )

        embedding, attention_weights = (
            self.attention(fused_features)
        )

        projection = self.projection_head(
            embedding
        )

        reconstructed_opencv = (
            self.opencv_reconstruction(embedding)
        )

        return {
            "embedding": embedding,
            "attention_weights": attention_weights,
            "projection": projection,
            "reconstructed_opencv": reconstructed_opencv
        }


model = CachedFusionModel(model_template)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.to(device)
model.eval()

print("Cached fusion model loaded")


# ============================================================
# LOAD CACHED CLEAN RESNET FEATURES
# ============================================================

cached = torch.load(
    FEATURE_FILE,
    map_location="cpu",
    weights_only=False
)

cached_image_features = (
    cached["image_features"].float()
)

cached_opencv_features = (
    cached["opencv_features"].float()
)

cached_sample_ids = cached["sample_ids"]

print(
    "Cached image features:",
    cached_image_features.shape
)


# ============================================================
# CLEAN REFERENCE EMBEDDINGS
# ============================================================

with torch.no_grad():

    normalized_opencv = (
        cached_opencv_features - feature_mean
    ) / feature_std

    clean_outputs = model(
        cached_image_features,
        normalized_opencv
    )

    clean_embeddings = (
        clean_outputs["embedding"]
    )


# ============================================================
# REFERENCE EMBEDDING
# ============================================================

reference_embedding = (
    clean_embeddings.mean(dim=0)
)

print("Reference embedding calculated")


# ============================================================
# BUILD FORM LOOKUP
# ============================================================

clean_lookup = {}

for index, row in metadata.iterrows():

    form_id = str(row["form_id"])

    image_path = (
        PROJECT_ROOT / row["image_path"]
    )

    opencv_features = torch.tensor(
        [
            row["skew"],
            row["baseline_deviation"],
            row["word_spacing_cv"],
            row["character_height_cv"],
            row["average_word_height"],
            row["average_word_width"],
            row["stroke_density"],
            row["slant_angle"],
            row["writing_area"],
        ],
        dtype=torch.float32
    )

    clean_lookup[form_id] = {
        "sample_id": str(row["sample_id"]),
        "image_path": image_path,
        "opencv_features": opencv_features,
    }


print(
    "Clean forms available:",
    len(clean_lookup)
)


# ============================================================
# RESNET50 FOR SYNTHETIC IMAGES
# ============================================================

weights = ResNet50_Weights.DEFAULT

resnet = resnet50(weights=weights)

resnet.fc = torch.nn.Identity()

resnet.to(device)
resnet.eval()

print("ResNet50 loaded")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
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

    image = (
        torch.from_numpy(image)
        .float()
        / 255.0
    )

    image = image.permute(2, 0, 1)

    return image


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def get_image_feature(image_path):

    image = load_image(image_path)

    image = image.unsqueeze(0).to(device)

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32
    ).view(1, 3, 1, 1)

    image = (
        image - mean
    ) / std

    with torch.no_grad():

        features = resnet(image)

    return features.squeeze(0)


# ============================================================
# CALCULATE CLEAN SCORE
# ============================================================

clean_scores = {}

print("Calculating clean scores...")


with torch.no_grad():

    for index, row in metadata.iterrows():

        form_id = str(row["form_id"])

        # Find matching cached sample
        matches = [
            i
            for i, sid in enumerate(cached_sample_ids)
            if str(sid) == str(row["sample_id"])
        ]

        if len(matches) == 0:
            continue

        cached_index = matches[0]

        embedding = (
            clean_embeddings[cached_index]
        )

        score = torch.norm(
            embedding - reference_embedding
        ).item()

        clean_scores[form_id] = score


print(
    "Clean scores calculated:",
    len(clean_scores)
)


# ============================================================
# SYNTHETIC SCORES
# ============================================================

results = []

print("\nProcessing synthetic anomalies...")


with torch.no_grad():

    for index, row in synthetic.iterrows():

        form_id = row["clean_form_id"]

        if form_id not in clean_lookup:

            print(
                "WARNING: form not found:",
                form_id
            )

            continue

        info = clean_lookup[form_id]

        if form_id not in clean_scores:

            print(
                "WARNING: clean score missing:",
                form_id
            )

            continue


        # ----------------------------------------------------
        # Synthetic image path
        # ----------------------------------------------------

        synthetic_image = (
            PROJECT_ROOT
            / row["synthetic_image"]
        )

        if not synthetic_image.exists():

            synthetic_image = (
                SYNTHETIC_IMAGE_DIR
                / Path(
                    row["synthetic_image"]
                ).name
            )


        if not synthetic_image.exists():

            print(
                "WARNING: synthetic image missing:",
                synthetic_image
            )

            continue


        # ----------------------------------------------------
        # ResNet feature
        # ----------------------------------------------------

        image_feature = (
            get_image_feature(
                synthetic_image
            )
        )


        # ----------------------------------------------------
        # ORIGINAL CLEAN OpenCV FEATURES
        #
        # IMPORTANT:
        # We intentionally keep these unchanged.
        # Therefore the benchmark tests the effect
        # of image corruption only.
        # ----------------------------------------------------

        opencv_feature = (
            info["opencv_features"]
        )

        normalized_opencv = (
            opencv_feature
            - feature_mean
        ) / feature_std


        # ----------------------------------------------------
        # Synthetic embedding
        # ----------------------------------------------------

        output = model(
            image_feature.unsqueeze(0),
            normalized_opencv.unsqueeze(0)
        )

        embedding = (
            output["embedding"]
            .squeeze(0)
        )


        # ----------------------------------------------------
        # Synthetic anomaly score
        # ----------------------------------------------------

        synthetic_score = torch.norm(
            embedding - reference_embedding
        ).item()


        # ----------------------------------------------------
        # Clean score
        # ----------------------------------------------------

        clean_score = (
            clean_scores[form_id]
        )


        # ----------------------------------------------------
        # Score change
        # ----------------------------------------------------

        score_change = (
            synthetic_score
            - clean_score
        )

        relative_change = (
            score_change
            / (clean_score + 1e-8)
        )


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "sample_id":
                row["sample_id"],

            "source_form":
                row["source_form"],

            "clean_form_id":
                form_id,

            "anomaly_type":
                row["anomaly_type"],

            "clean_score":
                clean_score,

            "synthetic_score":
                synthetic_score,

            "score_change":
                score_change,

            "relative_change":
                relative_change,

        })


        if (index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/{len(synthetic)}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print("PAIRED CLEAN VS SYNTHETIC RESULTS")
print("==========================================")

print(
    "Pairs:",
    len(results_df)
)


print("\nMean score change by anomaly type:")

summary = (
    results_df
    .groupby("anomaly_type")["score_change"]
    .agg([
        "count",
        "mean",
        "std",
        "min",
        "max"
    ])
)

print(summary)


print(
    "\nMean relative change by anomaly type:"
)

relative_summary = (
    results_df
    .groupby("anomaly_type")[
        "relative_change"
    ]
    .mean()
)

print(relative_summary)


print(
    "\nOverall mean clean score:",
    results_df["clean_score"].mean()
)

print(
    "Overall mean synthetic score:",
    results_df["synthetic_score"].mean()
)

print(
    "Overall mean score change:",
    results_df["score_change"].mean()
)


print("\nSaved to:")

print(OUTPUT_FILE)

