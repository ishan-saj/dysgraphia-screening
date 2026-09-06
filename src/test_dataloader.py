import torch
from torch.utils.data import DataLoader

from dataset import DysgraphiaDataset


# --------------------------------------------------
# Create dataset
# --------------------------------------------------

dataset = DysgraphiaDataset(
    metadata_csv="../data/metadata/metadata_all.csv",
    image_root="../data/raw/IAM/images",
)


# --------------------------------------------------
# Display dataset size
# --------------------------------------------------

print("\nDataset size:")
print(len(dataset))


# --------------------------------------------------
# Create DataLoader
# --------------------------------------------------

dataloader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    num_workers=0,
)


# --------------------------------------------------
# Get one batch
# --------------------------------------------------

batch = next(iter(dataloader))


# --------------------------------------------------
# Display batch information
# --------------------------------------------------

print("\nBatch information:")

print("\nImage batch:")
print(batch["image"].shape)

print("\nOpenCV feature batch:")
print(batch["opencv_features"].shape)

print("\nInk mask batch:")
print(batch["ink_mask"].shape)

print("\nSample IDs:")
print(batch["sample_id"])

print("\nImage paths:")
for path in batch["image_path"]:
    print(path)