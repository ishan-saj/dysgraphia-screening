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
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    def forward(self, x):

        return (x - self.mean) / self.std


# ============================================================
# OpenCV Feature Encoder
# ============================================================

class OpenCVEncoder(nn.Module):

    def __init__(
        self,
        input_dim=9,
        output_dim=64
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                32
            ),

            nn.ReLU(),

            nn.LayerNorm(32),

            nn.Linear(
                32,
                output_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.2)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# Feature Attention
# ============================================================

class FeatureAttention(nn.Module):

    def __init__(
        self,
        input_dim=2112,
        embedding_dim=256
    ):
        super().__init__()

        # Attention network

        self.attention = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                input_dim
            )
        )

        # Convert attended 2112 features
        # into a 256-dimensional embedding

        self.projection = nn.Sequential(

            nn.Linear(
                input_dim,
                embedding_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.2)
        )

    def forward(self, x):

        # Calculate attention scores

        attention_scores = self.attention(x)

        # Convert scores to weights

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # Apply attention

        attended_features = (
            x * attention_weights
        )

        # Create final embedding

        embedding = self.projection(
            attended_features
        )

        return (
            embedding,
            attention_weights
        )


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
        # ImageNet normalization
        # ----------------------------------------------------

        self.normalize = ImageNetNormalize()

        # ----------------------------------------------------
        # ResNet50
        # ----------------------------------------------------

        if pretrained:

            weights = ResNet50_Weights.DEFAULT

        else:

            weights = None

        self.resnet = resnet50(
            weights=weights
        )

        # Remove ImageNet classifier

        self.resnet.fc = nn.Identity()

        self.image_feature_dim = 2048

        # ----------------------------------------------------
        # Optional backbone freezing
        # ----------------------------------------------------

        if freeze_backbone:

            for parameter in self.resnet.parameters():

                parameter.requires_grad = False

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

        self.fusion_dim = (
            self.image_feature_dim + 64
        )

        # 2048 + 64 = 2112

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = FeatureAttention(
            input_dim=self.fusion_dim,
            embedding_dim=256
        )

        # ----------------------------------------------------
        # Self-supervised projection head
        # ----------------------------------------------------

        self.projection_head = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64
            )
        )

        # ----------------------------------------------------
        # OpenCV reconstruction head
        # ----------------------------------------------------
        #
        # 256-dimensional embedding
        #        ↓
        # 64-dimensional hidden layer
        #        ↓
        # 9 original OpenCV features
        #
        # ----------------------------------------------------

        self.opencv_reconstruction = nn.Sequential(

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                9
            )
        )

    # ========================================================
    # Forward Pass
    # ========================================================

    def forward(
        self,
        image,
        opencv_features
    ):

        # ----------------------------------------------------
        # IMAGE BRANCH
        # ----------------------------------------------------

        image = self.normalize(image)

        image_features = self.resnet(
            image
        )

        # Shape:
        # [batch_size, 2048]

        # ----------------------------------------------------
        # OPENCV BRANCH
        # ----------------------------------------------------

        encoded_opencv = self.opencv_encoder(
            opencv_features
        )

        # Shape:
        # [batch_size, 64]

        # ----------------------------------------------------
        # FEATURE FUSION
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                image_features,
                encoded_opencv
            ],
            dim=1
        )

        # Shape:
        # [batch_size, 2112]

        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        embedding, attention_weights = self.attention(
            fused_features
        )

        # Embedding:
        # [batch_size, 256]

        # Attention:
        # [batch_size, 2112]

        # ----------------------------------------------------
        # PROJECTION HEAD
        # ----------------------------------------------------

        projection = self.projection_head(
            embedding
        )

        # Shape:
        # [batch_size, 64]

        # ----------------------------------------------------
        # OPENCV RECONSTRUCTION
        # ----------------------------------------------------

        reconstructed_opencv = (
            self.opencv_reconstruction(
                embedding
            )
        )

        # Shape:
        # [batch_size, 9]

        # ----------------------------------------------------
        # RETURN ALL OUTPUTS
        # ----------------------------------------------------

        return {

            "image_features":
                image_features,

            "opencv_features":
                encoded_opencv,

            "fused_features":
                fused_features,

            "embedding":
                embedding,

            "projection":
                projection,

            "reconstructed_opencv":
                reconstructed_opencv,

            "attention_weights":
                attention_weights
        }