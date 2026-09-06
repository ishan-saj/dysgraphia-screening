import torch
from torch.utils.data import DataLoader

from dataset import DysgraphiaDataset
from model import DysgraphiaModel


# ============================================================
# Load Dataset
# ============================================================

dataset = DysgraphiaDataset(
    metadata_csv="../data/metadata/metadata_all.csv",
    image_root="../data/raw/IAM/images",
)


print("\nDataset size:")
print(len(dataset))


# ============================================================
# DataLoader
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Get Batch
# ============================================================

batch = next(iter(dataloader))


# ============================================================
# Create Model
# ============================================================

model = DysgraphiaModel(
    pretrained=True,
    freeze_backbone=False
)


# ============================================================
# Evaluation Mode
# ============================================================

model.eval()


# ============================================================
# Forward Pass
# ============================================================

with torch.no_grad():

    output = model(
        batch["image"],
        batch["opencv_features"]
    )


# ============================================================
# Display Results
# ============================================================

print("\n================ MODEL OUTPUT ================")

print("\nImage features:")
print(output["image_features"].shape)

print("\nOpenCV encoded features:")
print(output["opencv_features"].shape)

print("\nFused features:")
print(output["fused_features"].shape)

print("\nFinal embedding:")
print(output["embedding"].shape)

print("\nAttention weights:")
print(output["attention_weights"].shape)

print("\n================================================")
print("MODEL FORWARD PASS SUCCESSFUL")
print("================================================")