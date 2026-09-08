from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from model import DysgraphiaModel


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_ID = "p06-052"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "metadata" / "resnet_features.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "best_dysgraphia_encoder.pth"

IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "IAM" / "images"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / f"gradcam_anomaly_{SAMPLE_ID}.png"
)

DEVICE = torch.device("cpu")


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(sample_id):
    matches = list(IMAGE_ROOT.rglob(f"{sample_id}.png"))

    if not matches:
        matches = list(IMAGE_ROOT.rglob(f"{sample_id}.jpg"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find image for sample: {sample_id}"
        )

    return matches[0]


# ============================================================
# LOAD OPENCV FEATURES
# ============================================================

def load_opencv_features(sample_id):
    df = pd.read_csv(METADATA_PATH)

    feature_names = [
        "skew",
        "baseline_deviation",
        "word_spacing_cv",
        "character_height_cv",
        "average_word_height",
        "average_word_width",
        "stroke_density",
        "slant_angle",
        "writing_area",
    ]

    row = df[df["sample_id"].astype(str) == sample_id]

    if row.empty:
        raise ValueError(
            f"Sample {sample_id} was not found in metadata_all.csv"
        )

    features = (
        row[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values[0]
    )

    return torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0), feature_names


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # Convert grayscale to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    tensor = torch.from_numpy(
        rgb.transpose(2, 0, 1)
    ).float() / 255.0

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# LOAD CACHED FEATURES
# ============================================================

def load_reference_embeddings(model):
    cached = torch.load(
        CACHE_PATH,
        map_location=DEVICE
    )

    image_features = cached["image_features"].float()
    opencv_features = cached["opencv_features"].float()

    sample_ids = [
        str(x) for x in cached["sample_ids"]
    ]

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    feature_mean = checkpoint["feature_mean"].float()
    feature_std = checkpoint["feature_std"].float()

    # Normalize OpenCV features exactly as during training
    opencv_features = (
        opencv_features - feature_mean
    ) / (feature_std + 1e-8)

    with torch.no_grad():

        encoded_opencv = model.opencv_encoder(
            opencv_features
        )

        fused = torch.cat(
            [image_features, encoded_opencv],
            dim=1
        )

        embeddings, _ = model.attention(
            fused
        )

    # Reference = mean embedding of all samples
    reference_embedding = embeddings.mean(
        dim=0,
        keepdim=True
    )

    return (
        reference_embedding,
        feature_mean,
        feature_std,
        sample_ids
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(
            self.forward_hook
        )

    def forward_hook(self, module, input, output):

        self.activations = output

        # Capture gradient directly from the activation tensor
        if output.requires_grad:
            output.register_hook(
                self.save_gradient
            )

    def save_gradient(self, gradient):

        self.gradients = gradient

    def generate(
        self,
        image,
        opencv_features,
        reference_embedding
    ):

        self.model.zero_grad()

        output = self.model(
            image,
            opencv_features
        )

        embedding = output["embedding"]

        # Actual anomaly objective:
        # distance from the reference embedding
        anomaly_score = torch.norm(
            embedding - reference_embedding,
            p=2,
            dim=1
        )

        # Backpropagate anomaly score
        anomaly_score.sum().backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations
        gradients = self.gradients

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature maps
        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Keep positive influence
        cam = F.relu(cam)

        # Resize to input image
        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = (
            cam
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # Normalize
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam, anomaly_score.item()

# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(image_path, cam):

    original = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    original = cv2.resize(
        original,
        (224, 224)
    )

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return overlay


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("Grad-CAM for Actual Anomaly Score")
    print("=" * 60)

    print(f"\nSample ID: {SAMPLE_ID}")
    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path = find_image(SAMPLE_ID)

    print("\nImage found:")
    print(image_path)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading ResNet50 + fusion model...")

    model = DysgraphiaModel(
        pretrained=True,
        freeze_backbone=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # Only load trained fusion layers.
    # ResNet weights remain ImageNet-pretrained.
    model_state = checkpoint["model_state_dict"]

    compatible_state = {
        key: value
        for key, value in model_state.items()
        if key in model.state_dict()
        and model.state_dict()[key].shape == value.shape
    }

    model.load_state_dict(
        compatible_state,
        strict=False
    )

    model.to(DEVICE)
    model.eval()
    # Enable gradients for Grad-CAM
    for parameter in model.resnet.parameters():
      parameter.requires_grad = True

      
    # --------------------------------------------------------
    # OpenCV feature
    # --------------------------------------------------------

    opencv_features, feature_names = (
        load_opencv_features(SAMPLE_ID)
    )

    feature_mean = checkpoint[
        "feature_mean"
    ].float()

    feature_std = checkpoint[
        "feature_std"
    ].float()

    opencv_features = (
        opencv_features - feature_mean
    ) / (feature_std + 1e-8)

    opencv_features = opencv_features.to(DEVICE)

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    print("\nLoading image...")

    image = preprocess_image(
        image_path
    ).to(DEVICE)

    # --------------------------------------------------------
    # Reference embedding
    # --------------------------------------------------------

    print(
        "\nCalculating reference embedding "
        "from all 1539 samples..."
    )

    reference_embedding, _, _, sample_ids = (
        load_reference_embeddings(model)
    )

    print(
        f"Reference embedding shape: "
        f"{reference_embedding.shape}"
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    target_layer = model.resnet.layer4[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    print(
        "\nCalculating gradients "
        "for actual anomaly score..."
    )

    cam, anomaly_score = gradcam.generate(
        image,
        opencv_features,
        reference_embedding
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        image_path,
        cam
    )

    cv2.imwrite(
        str(OUTPUT_PATH),
        overlay
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Grad-CAM completed.")
    print("=" * 60)

    print(f"\nSample ID:")
    print(SAMPLE_ID)

    print(f"\nAnomaly score:")
    print(f"{anomaly_score:.6f}")

    print("\nInput image:")
    print(image_path)

    print("\nGrad-CAM saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()