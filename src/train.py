import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import DysgraphiaModel


# ============================================================
# Configuration
# ============================================================

FEATURE_FILE = "../data/metadata/resnet_features.pt"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

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
            "image_features": self.image_features[index],
            "opencv_features": self.opencv_features[index],
            "sample_id": self.sample_ids[index],
            "image_path": self.image_paths[index],
        }


# ============================================================
# Load Dataset
# ============================================================

print(f"Using device: {DEVICE}")

dataset = CachedFeatureDataset(FEATURE_FILE)

print(f"\nTotal usable samples: {len(dataset)}")


# ============================================================
# Train / Validation Split
# ============================================================

validation_size = int(
    len(dataset) * VALIDATION_SPLIT
)

training_size = len(dataset) - validation_size

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = random_split(
    dataset,
    [training_size, validation_size],
    generator=generator
)


# ============================================================
# Calculate normalization statistics
# ONLY from training data
# ============================================================

train_indices = train_dataset.indices

train_opencv_features = dataset.opencv_features[
    train_indices
]

feature_mean = train_opencv_features.mean(
    dim=0
)

feature_std = train_opencv_features.std(
    dim=0
)

# Prevent division by zero
feature_std[feature_std < 1e-8] = 1.0


print("\nOpenCV feature normalization:")
print("-" * 50)

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

for i, name in enumerate(feature_names):

    print(
        f"{name:25s} "
        f"mean={feature_mean[i]:.4f} "
        f"std={feature_std[i]:.4f}"
    )


# ============================================================
# Normalize OpenCV features
# ============================================================

dataset.opencv_features = (
    dataset.opencv_features - feature_mean
) / feature_std


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print(f"\nTraining samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(validation_loader)}")


# ============================================================
# Model
# ============================================================

print("\nCreating fusion model...")

original_model = DysgraphiaModel(
    pretrained=False,
    freeze_backbone=True
)


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
            "image_features": image_features,
            "opencv_features": encoded_opencv,
            "fused_features": fused_features,
            "embedding": embedding,
            "projection": projection,
            "reconstructed_opencv":
                reconstructed_opencv,
            "attention_weights":
                attention_weights,
        }


model = CachedFusionModel(
    original_model
).to(DEVICE)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# Loss
# ============================================================

reconstruction_loss = nn.MSELoss()


# ============================================================
# Training
# ============================================================

best_validation_loss = float("inf")

print("\n")
print("=" * 50)
print("Starting training")
print("=" * 50)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0

    for batch in train_loader:

        image_features = batch[
            "image_features"
        ].to(DEVICE)

        opencv_features = batch[
            "opencv_features"
        ].to(DEVICE)

        optimizer.zero_grad()

        output = model(
            image_features,
            opencv_features
        )

        loss = reconstruction_loss(
            output["reconstructed_opencv"],
            opencv_features
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()


    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    total_validation_loss = 0.0

    with torch.no_grad():

        for batch in validation_loader:

            image_features = batch[
                "image_features"
            ].to(DEVICE)

            opencv_features = batch[
                "opencv_features"
            ].to(DEVICE)

            output = model(
                image_features,
                opencv_features
            )

            loss = reconstruction_loss(
                output["reconstructed_opencv"],
                opencv_features
            )

            total_validation_loss += loss.item()


    average_validation_loss = (
        total_validation_loss /
        len(validation_loader)
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {average_train_loss:.6f} | "
        f"Validation Loss: "
        f"{average_validation_loss:.6f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if average_validation_loss < best_validation_loss:

        best_validation_loss = (
            average_validation_loss
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "feature_names": feature_names,
            },
            "../best_dysgraphia_encoder.pth"
        )

        print(
            "  -> Best model saved."
        )


# ============================================================
# Finished
# ============================================================

print("\n")
print("=" * 50)
print("Training completed.")
print("=" * 50)

print(
    f"Best validation loss: "
    f"{best_validation_loss:.6f}"
)