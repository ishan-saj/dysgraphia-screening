import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # type: ignore[reportMissingImports]
import torch.nn as nn  # type: ignore[reportMissingImports]


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "resnet_features.pt"
)

CHECKPOINT_FILE = (
    PROJECT_ROOT
    / "best_dysgraphia_encoder.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
)

DEVICE = torch.device("cpu")


# ============================================================
# FEATURE NAMES
# EXACT ORDER USED DURING TRAINING
# ============================================================

FEATURE_NAMES = [
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


# ============================================================
# MODEL ARCHITECTURE
# MUST MATCH save_paired_embeddings.py
# ============================================================

class OpenCVEncoder(nn.Module):
    """
    Encodes the 9 OpenCV handwriting features
    into a 64-dimensional representation.
    """

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
    """
    Attention + projection layer.

    Input:
        2048 ResNet features
        +
        64 OpenCV features
        =
        2112 features

    Output:
        256-dimensional embedding
    """

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

        attended_features = x * attention_weights

        embedding = self.projection(
            attended_features
        )

        return embedding


class CachedFusionModel(nn.Module):
    """
    Exact fusion model used by the project.
    """

    def __init__(self):
        super().__init__()

        self.opencv_encoder = OpenCVEncoder()

        self.attention = FeatureAttention()

    def forward(
        self,
        image_features,
        opencv_features
    ):

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
# LOAD DATA
# ============================================================

def load_project_data():

    print("\nLoading cached features...")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"\nFeature file not found:\n{FEATURE_FILE}"
        )

    cached = torch.load(
        FEATURE_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    image_features = cached[
        "image_features"
    ].float()

    opencv_features = cached[
        "opencv_features"
    ].float()

    sample_ids = [
        str(x)
        for x in cached["sample_ids"]
    ]

    print(
        f"Loaded {len(sample_ids)} samples."
    )

    return (
        image_features,
        opencv_features,
        sample_ids
    )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("\nLoading trained model...")

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{CHECKPOINT_FILE}"
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # These are the SAME mean/std values calculated
    # from the training set in train.py.
    # --------------------------------------------------------

    feature_mean = checkpoint[
        "feature_mean"
    ].float()

    feature_std = checkpoint[
        "feature_std"
    ].float()

    model = CachedFusionModel()

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False
    )

    model.eval()

    print("Model loaded successfully.")

    print(
        f"OpenCV feature mean shape: "
        f"{tuple(feature_mean.shape)}"
    )

    print(
        f"OpenCV feature std shape: "
        f"{tuple(feature_std.shape)}"
    )

    return (
        model,
        feature_mean,
        feature_std
    )


# ============================================================
# FIND SAMPLE
# ============================================================

def get_sample(
    sample_id,
    image_features,
    opencv_features,
    sample_ids
):

    if sample_id not in sample_ids:

        print("\nERROR: Sample ID not found.")

        print("\nAvailable examples:")

        for sid in sample_ids[:20]:
            print("  ", sid)

        raise ValueError(
            f"\nSample '{sample_id}' does not exist "
            f"in resnet_features.pt."
        )

    index = sample_ids.index(
        sample_id
    )

    image_feature = image_features[
        index
    ].float()

    opencv_feature = opencv_features[
        index
    ].float()

    return (
        index,
        image_feature,
        opencv_feature
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_opencv_features(
    features,
    feature_mean,
    feature_std
):

    return (
        features - feature_mean
    ) / (
        feature_std + 1e-8
    )


# ============================================================
# COMPUTE EMBEDDING
# ============================================================

def compute_embedding(
    model,
    image_features,
    opencv_features,
    feature_mean,
    feature_std
):

    normalized_opencv = (
        normalize_opencv_features(
            opencv_features,
            feature_mean,
            feature_std
        )
    )

    with torch.no_grad():

        embedding = model(
            image_features,
            normalized_opencv
        )

    return embedding.squeeze(0)


# ============================================================
# FORM FEATURE ATTRIBUTION
# ============================================================

def calculate_feature_attribution(
    model,
    image_feature,
    opencv_feature,
    feature_mean,
    feature_std
):

    # --------------------------------------------------------
    # Make sure tensors have batch dimension.
    # --------------------------------------------------------

    image_feature = (
        image_feature
        .float()
        .unsqueeze(0)
    )

    opencv_feature = (
        opencv_feature
        .float()
    )

    feature_mean = (
        feature_mean
        .float()
    )

    feature_std = (
        feature_std
        .float()
    )

    # --------------------------------------------------------
    # BASELINE
    #
    # This is the original sample representation.
    # --------------------------------------------------------

    baseline_embedding = compute_embedding(
        model=model,
        image_features=image_feature,
        opencv_features=opencv_feature.unsqueeze(0),
        feature_mean=feature_mean,
        feature_std=feature_std
    )

    results = []

    # --------------------------------------------------------
    # ONE-FEATURE-AT-A-TIME PERTURBATION
    # --------------------------------------------------------

    for i, feature_name in enumerate(
        FEATURE_NAMES
    ):

        modified_features = (
            opencv_feature.clone()
        )

        original_value = (
            modified_features[i].item()
        )

        reference_value = (
            feature_mean[i].item()
        )

        # ----------------------------------------------------
        # Replace only this feature with its training
        # reference value.
        #
        # All other features remain unchanged.
        # ----------------------------------------------------

        modified_features[i] = (
            feature_mean[i]
        )

        modified_embedding = compute_embedding(
            model=model,
            image_features=image_feature,
            opencv_features=modified_features.unsqueeze(0),
            feature_mean=feature_mean,
            feature_std=feature_std
        )

        # ----------------------------------------------------
        # Measure how much the 256-D representation changed.
        # ----------------------------------------------------

        embedding_change = torch.norm(
            baseline_embedding
            - modified_embedding,
            p=2
        ).item()

        results.append(
            {
                "feature": feature_name,
                "original_value": original_value,
                "reference_value": reference_value,
                "embedding_change": embedding_change
            }
        )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Total measured change
    # --------------------------------------------------------

    total_change = (
        results_df[
            "embedding_change"
        ].sum()
    )

    if total_change > 1e-12:

        results_df[
            "relative_contribution_pct"
        ] = (
            results_df[
                "embedding_change"
            ]
            / total_change
            * 100.0
        )

    else:

        results_df[
            "relative_contribution_pct"
        ] = 0.0

    # --------------------------------------------------------
    # Rank features
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            "embedding_change",
            ascending=False
        )
        .reset_index(drop=True)
    )

    results_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(results_df) + 1
        )
    )

    return (
        results_df,
        baseline_embedding,
        total_change
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    sample_id,
    results_df
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / f"{sample_id}_form_feature_attribution.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    return output_file


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    sample_id,
    results_df,
    total_change
):

    print("\n")
    print("=" * 78)
    print("FORM-LEVEL FEATURE ATTRIBUTION")
    print("=" * 78)

    print(
        f"Sample: {sample_id}"
    )

    print(
        f"Total measured embedding change: "
        f"{total_change:.6f}"
    )

    print("-" * 78)

    print(
        f"{'Rank':<6}"
        f"{'Feature':<28}"
        f"{'Original':>12}"
        f"{'Reference':>12}"
        f"{'Change':>12}"
        f"{'Contribution':>14}"
    )

    print("-" * 78)

    for _, row in results_df.iterrows():

        print(
            f"{int(row['rank']):<6}"
            f"{row['feature']:<28}"
            f"{row['original_value']:>12.5f}"
            f"{row['reference_value']:>12.5f}"
            f"{row['embedding_change']:>12.6f}"
            f"{row['relative_contribution_pct']:>13.2f}%"
        )

    print("-" * 78)

    # --------------------------------------------------------
    # Top feature
    # --------------------------------------------------------

    top = results_df.iloc[0]

    print(
        "\nMost influential form-level feature:"
    )

    print(
        f"  {top['feature']}"
    )

    print(
        f"  Embedding change: "
        f"{top['embedding_change']:.6f}"
    )

    print(
        f"  Relative contribution: "
        f"{top['relative_contribution_pct']:.2f}%"
    )

    print(
        "\nInterpretation:"
    )

    print(
        f"Replacing '{top['feature']}' with its "
        f"training reference value produced the "
        f"largest change in the learned 256-D "
        f"representation among the nine form-level "
        f"features."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a representation-sensitivity "
        "analysis. The percentage is NOT a "
        "probability that the feature caused "
        "dysgraphia or the anomaly."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # SAMPLE ID
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python form_feature_attribution.py SAMPLE_ID"
        )

        print(
            "\nExample:"
        )

        print(
            "python form_feature_attribution.py p06-052"
        )

        print(
            "python form_feature_attribution.py a01-063"
        )

        print(
            "python form_feature_attribution.py j06-051"
        )

        sys.exit(1)

    sample_id = sys.argv[1]

    print("\n")
    print("=" * 78)
    print("FORM-LEVEL FEATURE ATTRIBUTION ANALYSIS")
    print("=" * 78)

    print(
        f"Sample requested: {sample_id}"
    )

    # --------------------------------------------------------
    # Load cached features
    # --------------------------------------------------------

    (
        image_features,
        opencv_features,
        sample_ids
    ) = load_project_data()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        feature_mean,
        feature_std
    ) = load_model()

    # --------------------------------------------------------
    # Get exact sample
    # --------------------------------------------------------

    (
        sample_index,
        image_feature,
        opencv_feature
    ) = get_sample(
        sample_id=sample_id,
        image_features=image_features,
        opencv_features=opencv_features,
        sample_ids=sample_ids
    )

    print(
        f"\nSample index: {sample_index}"
    )

    print(
        f"Image feature dimension: "
        f"{tuple(image_feature.shape)}"
    )

    print(
        f"OpenCV feature dimension: "
        f"{tuple(opencv_feature.shape)}"
    )

    # --------------------------------------------------------
    # Print original feature values
    # --------------------------------------------------------

    print(
        "\nOriginal OpenCV features:"
    )

    for name, value in zip(
        FEATURE_NAMES,
        opencv_feature.tolist()
    ):

        print(
            f"  {name:<28}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Calculate attribution
    # --------------------------------------------------------

    (
        results_df,
        baseline_embedding,
        total_change
    ) = calculate_feature_attribution(
        model=model,
        image_feature=image_feature,
        opencv_feature=opencv_feature,
        feature_mean=feature_mean,
        feature_std=feature_std
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = save_results(
        sample_id=sample_id,
        results_df=results_df
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        sample_id=sample_id,
        results_df=results_df,
        total_change=total_change
    )

    # --------------------------------------------------------
    # Final output path
    # --------------------------------------------------------

    print(
        "\nCSV saved to:"
    )

    print(
        f"  {output_file}"
    )

    print(
        "\nAnalysis completed successfully."
    )

    print("=" * 78)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()