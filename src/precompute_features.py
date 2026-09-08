import os
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

from dataset import DysgraphiaDataset


# ============================================================
# Configuration
# ============================================================

METADATA_CSV = "../data/metadata/metadata_all.csv"
IMAGE_ROOT = "../data/raw/IAM/images"

OUTPUT_FILE = "../data/metadata/resnet_features.pt"

BATCH_SIZE = 16
NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ============================================================
# Dataset
# ============================================================

dataset = DysgraphiaDataset(
    metadata_csv=METADATA_CSV,
    image_root=IMAGE_ROOT,
)

print(f"\nTotal samples: {len(dataset)}")


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# Load pretrained ResNet50
# ============================================================

print("\nLoading ResNet50...")

weights = ResNet50_Weights.DEFAULT

resnet = resnet50(weights=weights)

# Remove classification layer
resnet.fc = torch.nn.Identity()

resnet = resnet.to(device)

# Freeze ResNet
resnet.eval()

for parameter in resnet.parameters():
    parameter.requires_grad = False


# ============================================================
# ImageNet normalization
# ============================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(1, 3, 1, 1)


# ============================================================
# Feature extraction
# ============================================================

all_image_features = []
all_opencv_features = []
all_sample_ids = []
all_image_paths = []

print("\nStarting feature extraction...")
print("=" * 50)


with torch.no_grad():

    for batch_idx, batch in enumerate(dataloader):

        images = batch["image"].to(device)

        # ImageNet normalization
        images = (images - mean) / std

        # ResNet50 feature extraction
        image_features = resnet(images)

        all_image_features.append(
            image_features.cpu()
        )

        all_opencv_features.append(
            batch["opencv_features"].cpu()
        )

        all_sample_ids.extend(
            batch["sample_id"]
        )

        all_image_paths.extend(
            batch["image_path"]
        )

        print(
            f"Batch {batch_idx + 1}/{len(dataloader)} "
            f"-> features: {image_features.shape}"
        )


# ============================================================
# Combine features
# ============================================================

image_features = torch.cat(
    all_image_features,
    dim=0
)

opencv_features = torch.cat(
    all_opencv_features,
    dim=0
)


# ============================================================
# Save
# ============================================================

feature_data = {
    "image_features": image_features,
    "opencv_features": opencv_features,
    "sample_ids": all_sample_ids,
    "image_paths": all_image_paths,
}


# Create directory if necessary
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


torch.save(
    feature_data,
    OUTPUT_FILE
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 50)
print("Feature extraction completed.")
print("=" * 50)

print(f"Image features shape: {image_features.shape}")
print(f"OpenCV features shape: {opencv_features.shape}")
print(f"Number of samples: {len(all_sample_ids)}")

print(f"\nSaved to:")
print(OUTPUT_FILE)