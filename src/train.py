import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import DysgraphiaDataset
from model import DysgraphiaModel


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 3
EPOCHS = 5
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# Load Dataset
# ============================================================

dataset = DysgraphiaDataset(
    metadata_csv="../data/metadata/metadata_all.csv",
    image_root="../data/raw/IAM/images",
)

print("\nTotal samples available:", len(dataset))


# ============================================================
# DataLoader
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

print("Number of batches:", len(dataloader))


# ============================================================
# Create Model
# ============================================================

model = DysgraphiaModel(
    pretrained=True,
    freeze_backbone=False
)

model = model.to(DEVICE)


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# Training
# ============================================================

print("\nStarting training...\n")

model.train()

for epoch in range(EPOCHS):

    total_embedding_loss = 0.0

    for batch_index, batch in enumerate(dataloader):

        # ----------------------------------------------------
        # Move data to device
        # ----------------------------------------------------

        images = batch["image"].to(DEVICE)

        opencv_features = batch[
            "opencv_features"
        ].to(DEVICE)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(
            images,
            opencv_features
        )

        embedding = output["embedding"]

        # ----------------------------------------------------
        # Temporary self-supervised loss
        #
        # This keeps the pipeline trainable without pretending
        # that IAM has clinical dysgraphia labels.
        # ----------------------------------------------------

        loss = torch.mean(
            torch.sum(
                embedding ** 2,
                dim=1
            )
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_embedding_loss += loss.item()

        # ----------------------------------------------------
        # Display progress
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Batch [{batch_index + 1}/{len(dataloader)}] "
            f"Loss: {loss.item():.4f}"
        )

    # --------------------------------------------------------
    # Epoch result
    # --------------------------------------------------------

    average_loss = (
        total_embedding_loss / len(dataloader)
    )

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"Average Loss: {average_loss:.4f}\n"
    )


# ============================================================
# Save Model
# ============================================================

torch.save(
    model.state_dict(),
    "../dysgraphia_model.pth"
)

print("Training completed.")
print("Model saved to: ../dysgraphia_model.pth")