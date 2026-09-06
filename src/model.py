import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# ImageNet Normalization
# ============================================================

class ImageNetNormalize(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        return (x - self.mean) / self.std


# ============================================================
# OpenCV Feature Encoder
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


# ============================================================
# Feature Attention
# ============================================================

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

        # Calculate attention weights
        attention_scores = self.attention(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # Apply attention
        attended_features = x * attention_weights

        # Create final embedding
        embedding = self.projection(attended_features)

        return embedding, attention_weights


# ============================================================
# Main Dysgraphia Model
# ============================================================

class DysgraphiaModel(nn.Module):

    def __init__(
        self,
        pretrained=True,
        freeze_backbone=False
    ):
        super().__init__()

        # ----------------------------------------------------
        # Image normalization
        # ----------------------------------------------------

        self.normalize = ImageNetNormalize()

        # ----------------------------------------------------
        # ResNet50
        # ----------------------------------------------------

        if pretrained:

            weights = ResNet50_Weights.DEFAULT

        else:

            weights = None

        self.resnet = resnet50(weights=weights)

        # Remove original ImageNet classifier
        self.resnet.fc = nn.Identity()

        # ----------------------------------------------------
        # Freeze ResNet if requested
        # ----------------------------------------------------

        if freeze_backbone:

            for parameter in self.resnet.parameters():
                parameter.requires_grad = False

        # ResNet50 output = 2048
        self.image_feature_dim = 2048

        # ----------------------------------------------------
        # OpenCV encoder
        # ----------------------------------------------------

        self.opencv_encoder = OpenCVEncoder(
            input_dim=9,
            output_dim=64
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.fusion_dim = 2048 + 64

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = FeatureAttention(
            input_dim=self.fusion_dim,
            embedding_dim=256
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(self, image, opencv_features):

        # ----------------------------------------------------
        # Image branch
        # ----------------------------------------------------

        image = self.normalize(image)

        image_features = self.resnet(image)

        # image_features shape:
        # [batch_size, 2048]

        # ----------------------------------------------------
        # OpenCV branch
        # ----------------------------------------------------

        opencv_features = self.opencv_encoder(
            opencv_features
        )

        # opencv_features shape:
        # [batch_size, 64]

        # ----------------------------------------------------
        # Feature Fusion
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                image_features,
                opencv_features
            ],
            dim=1
        )

        # fused_features:
        # [batch_size, 2112]

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        embedding, attention_weights = self.attention(
            fused_features
        )

        # embedding:
        # [batch_size, 256]

        return {
            "image_features": image_features,
            "opencv_features": opencv_features,
            "fused_features": fused_features,
            "embedding": embedding,
            "attention_weights": attention_weights
        }