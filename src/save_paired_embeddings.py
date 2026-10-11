from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from dataset import (
    load_handwriting_crop,
    resize_handwriting_crop,
    crop_to_resnet_tensor,
    extract_pixel_features,
    FORM_ID_MAP,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = PROJECT_ROOT / "data" / "metadata" / "resnet_features.pt"
SYNTHETIC_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_resnet_features.pt"
)
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
# IMAGE LOADING (SHARED CROP + INTER_AREA RESIZE)
# ============================================================

def load_crop_and_tensor(path, source_form=None, padding=30):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if source_form is None:
        raise ValueError(f"source_form must be provided for {path}")
    crop_gray = load_handwriting_crop(source_form, image, padding=padding)
    tensor_224 = crop_to_resnet_tensor(crop_gray, size=224)
    return crop_gray, tensor_224


def load_image(path, source_form=None, padding=30):
    _, tensor_224 = load_crop_and_tensor(path, source_form=source_form, padding=padding)
    return tensor_224


# ============================================================
# RESNET FEATURE
# ============================================================

def get_resnet_and_pixel_features(model, path, source_form=None):
    crop_gray, tensor_224 = load_crop_and_tensor(
        path,
        source_form=source_form
    )
    pixel_cv_9d = extract_pixel_features(crop_gray)
    image = tensor_224.unsqueeze(0)

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

    return feature.squeeze(0), torch.from_numpy(pixel_cv_9d).float()


if __name__ == "__main__":
    print("Loading cached features...")

    cached = torch.load(
        FEATURE_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    assert bool(cached.get("cropped", False)) is True, (
        f"Refusing to run: {FEATURE_FILE} does not have cropped=True!"
    )
    assert str(cached.get("interp", "")) == "INTER_AREA", (
        f"Refusing to run: {FEATURE_FILE} does not have interp='INTER_AREA'!"
    )

    clean_image_features = cached["image_features"].float()
    # Use pixel-extracted 9-D vectors for clean forms
    clean_opencv_features = cached["pixel_opencv_features"].float()

    sample_ids = [str(x) for x in cached["sample_ids"]]
    print("Clean samples:", len(sample_ids))

    clean_index_by_id = {
        sample_ids[i]: i
        for i in range(len(sample_ids))
    }

    print("\nLoading retrained pixel checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    feature_mean = checkpoint["feature_mean"].float()
    feature_std = checkpoint["feature_std"].float()

    model = CachedFusionModel()
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False
    )
    model.eval()
    print("Fusion model loaded.")

    print("\nCalculating clean embeddings...")
    normalized_opencv = (
        clean_opencv_features - feature_mean
    ) / (feature_std + 1e-8)

    with torch.no_grad():
        clean_embeddings = model(
            clean_image_features,
            normalized_opencv
        )

    clean_embeddings = clean_embeddings.cpu().numpy()
    print("Clean embeddings:", clean_embeddings.shape)

    print("\nLoading ResNet50...")
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval()
    print("ResNet50 loaded.")

    synthetic = pd.read_csv(SYNTHETIC_CSV)
    print("\nSynthetic samples:", len(synthetic))

    synthetic_embeddings = []
    synthetic_resnet_features = []
    synthetic_pixel_opencv_features = []
    synthetic_sample_ids = []
    synthetic_types = []
    source_forms = []
    source_indices = []

    for index, row in synthetic.iterrows():
        source_form = str(row["source_form"])
        source_form = FORM_ID_MAP.get(source_form, source_form)

        if source_form not in clean_index_by_id:
            print("Skipping unmatched form:", source_form)
            continue

        source_index = clean_index_by_id[source_form]
        image_path = PROJECT_ROOT / row["synthetic_image"]
        if not image_path.exists():
            image_path = (
                PROJECT_ROOT
                / "data"
                / "synthetic_form_anomalies"
                / "images"
                / Path(row["synthetic_image"]).name
            )

        # Extract BOTH 2048-D ResNet (INTER_AREA) AND 9-D pixel-derived OpenCV features from the corrupted crop itself!
        raw_feat, synth_cv_vec = get_resnet_and_pixel_features(
            resnet,
            image_path,
            source_form=source_form
        )
        synthetic_resnet_features.append(raw_feat.cpu())
        synthetic_pixel_opencv_features.append(synth_cv_vec.cpu())
        synthetic_sample_ids.append(str(row["sample_id"]))

        image_feature = raw_feat.unsqueeze(0)
        opencv_feature = (synth_cv_vec.unsqueeze(0) - feature_mean) / (feature_std + 1e-8)

        with torch.no_grad():
            embedding = model(image_feature, opencv_feature)

        embedding = embedding.squeeze(0).cpu().numpy()
        synthetic_embeddings.append(embedding)
        synthetic_types.append(row["anomaly_type"])
        source_forms.append(source_form)
        source_indices.append(source_index)

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(synthetic)}")

    synthetic_resnet_tensor = torch.stack(synthetic_resnet_features, dim=0)
    synthetic_pixel_cv_tensor = torch.stack(synthetic_pixel_opencv_features, dim=0)
    assert not torch.isnan(synthetic_resnet_tensor).any() and not torch.isinf(synthetic_resnet_tensor).any(), "NaN/Inf in synthetic_resnet_tensor!"
    assert not torch.isnan(synthetic_pixel_cv_tensor).any() and not torch.isinf(synthetic_pixel_cv_tensor).any(), "NaN/Inf in synthetic_pixel_cv_tensor!"

    torch.save(
        {
            "image_features": synthetic_resnet_tensor,
            "pixel_opencv_features": synthetic_pixel_cv_tensor,
            "opencv_features": synthetic_pixel_cv_tensor,
            "sample_ids": synthetic_sample_ids,
            "anomaly_types": list(synthetic_types),
            "source_forms": list(source_forms),
            "cropped": True,
            "crop_mode": "handwriting_bbox_p30",
            "interp": "INTER_AREA",
            "opencv_source": "pixel_extracted",
        },
        SYNTHETIC_FEATURE_FILE,
    )
    print(f"Saved cropped + INTER_AREA + pixel-OpenCV synthetic features to: {SYNTHETIC_FEATURE_FILE}")

    synthetic_embeddings = np.array(synthetic_embeddings)
    assert not np.isnan(clean_embeddings).any() and not np.isinf(clean_embeddings).any(), "NaN/Inf in clean_embeddings!"
    assert not np.isnan(synthetic_embeddings).any() and not np.isinf(synthetic_embeddings).any(), "NaN/Inf in synthetic_embeddings!"

    synthetic_types = np.array(synthetic_types)
    source_forms = np.array(source_forms)
    source_indices = np.array(source_indices, dtype=np.int64)

    np.savez(
        OUTPUT_FILE,
        clean_embeddings=clean_embeddings,
        synthetic_embeddings=synthetic_embeddings,
        clean_pixel_opencv=clean_opencv_features.numpy(),
        synthetic_pixel_opencv=synthetic_pixel_cv_tensor.numpy(),
        synthetic_types=synthetic_types,
        source_forms=source_forms,
        source_indices=source_indices,
        cropped=np.array(True),
        crop_mode=np.array("handwriting_bbox_p30"),
        interp=np.array("INTER_AREA"),
        opencv_source=np.array("pixel_extracted"),
    )

    print("\nSaved paired embeddings to:")
    print(OUTPUT_FILE)
    print("\nDone.")