import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # type: ignore[reportMissingImports]
import torch.nn as nn  # type: ignore[reportMissingImports]

from sklearn.neighbors import LocalOutlierFactor


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
# TRAINING / LOF SETTINGS
# MUST MATCH THE PROJECT
# ============================================================

RANDOM_SEED = 42

VALIDATION_RATIO = 0.20

LOF_N_NEIGHBORS = 20

LOF_CONTAMINATION = "auto"

LOF_METRIC = "euclidean"


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
# MUST MATCH TRAINING MODEL
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

    2048 ResNet features
    +
    64 OpenCV features
    =
    2112 fused features

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

    print(
        f"Image features: "
        f"{tuple(image_features.shape)}"
    )

    print(
        f"OpenCV features: "
        f"{tuple(opencv_features.shape)}"
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
    # These are the training-set mean/std values
    # calculated during train.py.
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

    # --------------------------------------------------------
    # VERY IMPORTANT
    #
    # Dropout must be disabled during attribution.
    # --------------------------------------------------------

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
# RECREATE TRAIN / VALIDATION SPLIT
# MUST MATCH train.py
# ============================================================

def create_training_indices(
    num_samples
):

    # --------------------------------------------------------
    # train.py:
    #
    # random_split(...,
    #     generator=torch.Generator().manual_seed(42))
    #
    # validation = 20%
    # --------------------------------------------------------

    validation_size = int(
        num_samples * VALIDATION_RATIO
    )

    train_size = (
        num_samples
        - validation_size
    )

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    indices = torch.randperm(
        num_samples,
        generator=generator
    )

    train_indices = indices[
        :train_size
    ]

    validation_indices = indices[
        train_size:
    ]

    print("\nRecreated project split:")

    print(
        f"Training samples: "
        f"{len(train_indices)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_indices)}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    return (
        train_indices,
        validation_indices
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
# COMPUTE ONE OR MANY EMBEDDINGS
# ============================================================

def compute_embeddings(
    model,
    image_features,
    opencv_features,
    feature_mean,
    feature_std
):

    normalized_opencv = normalize_opencv_features(
        opencv_features,
        feature_mean,
        feature_std
    )

    with torch.no_grad():

        embeddings = model(
            image_features,
            normalized_opencv
        )

    return embeddings


# ============================================================
# BUILD LOF REFERENCE MODEL
# ============================================================

def build_lof_reference(
    model,
    image_features,
    opencv_features,
    feature_mean,
    feature_std
):

    print("\nComputing all 256-D embeddings...")

    all_embeddings = compute_embeddings(
        model=model,
        image_features=image_features,
        opencv_features=opencv_features,
        feature_mean=feature_mean,
        feature_std=feature_std
    )

    print(
        f"Embedding matrix: "
        f"{tuple(all_embeddings.shape)}"
    )

    # --------------------------------------------------------
    # Recreate EXACT training/validation split.
    # --------------------------------------------------------

    (
        train_indices,
        validation_indices
    ) = create_training_indices(
        len(image_features)
    )

    train_embeddings = (
        all_embeddings[
            train_indices
        ]
        .cpu()
        .numpy()
    )

    validation_embeddings = (
        all_embeddings[
            validation_indices
        ]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # LOF is fitted ONLY on training embeddings.
    #
    # novelty=True allows scoring unseen validation
    # or perturbed samples.
    # --------------------------------------------------------

    print("\nFitting LOF reference model...")

    lof = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS,
        contamination=LOF_CONTAMINATION,
        metric=LOF_METRIC,
        novelty=True
    )

    lof.fit(
        train_embeddings
    )

    print(
        "LOF reference model fitted successfully."
    )

    print(
        f"LOF neighbours: "
        f"{LOF_N_NEIGHBORS}"
    )

    print(
        f"LOF metric: "
        f"{LOF_METRIC}"
    )

    return (
        lof,
        all_embeddings,
        train_indices,
        validation_indices
    )


# ============================================================
# GET SAMPLE
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

            print(
                "  ",
                sid
            )

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
# COMPUTE LOF ANOMALY SCORE
# ============================================================

def compute_lof_score(
    lof,
    embedding
):

    # --------------------------------------------------------
    # sklearn LOF:
    #
    # score_samples()
    #
    # More negative = more abnormal.
    #
    # Our project anomaly score uses:
    #
    # -score_samples()
    #
    # Therefore:
    # higher = more anomalous
    # --------------------------------------------------------

    embedding_np = (
        embedding
        .detach()
        .cpu()
        .numpy()
        .reshape(1, -1)
    )

    lof_score = -float(
        lof.score_samples(
            embedding_np
        )[0]
    )

    return lof_score


# ============================================================
# LOF FEATURE SENSITIVITY
# ============================================================

def calculate_lof_feature_sensitivity(
    model,
    lof,
    image_feature,
    opencv_feature,
    feature_mean,
    feature_std
):

    # --------------------------------------------------------
    # Prepare tensors.
    # --------------------------------------------------------

    image_feature_batch = (
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
    # ORIGINAL SAMPLE
    # --------------------------------------------------------

    baseline_embedding = compute_embeddings(
        model=model,
        image_features=image_feature_batch,
        opencv_features=opencv_feature.unsqueeze(0),
        feature_mean=feature_mean,
        feature_std=feature_std
    )

    original_lof_score = compute_lof_score(
        lof=lof,
        embedding=baseline_embedding[0]
    )

    print(
        f"\nOriginal LOF anomaly score: "
        f"{original_lof_score:.6f}"
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

        original_value = float(
            modified_features[i].item()
        )

        reference_value = float(
            feature_mean[i].item()
        )

        # ----------------------------------------------------
        # Replace ONLY this feature with the
        # training reference mean.
        #
        # All other features remain unchanged.
        # ----------------------------------------------------

        modified_features[i] = (
            feature_mean[i]
        )

        # ----------------------------------------------------
        # Recalculate the 256-D representation.
        # ----------------------------------------------------

        modified_embedding = compute_embeddings(
            model=model,
            image_features=image_feature_batch,
            opencv_features=modified_features.unsqueeze(0),
            feature_mean=feature_mean,
            feature_std=feature_std
        )

        # ----------------------------------------------------
        # Recalculate LOF score.
        # ----------------------------------------------------

        modified_lof_score = compute_lof_score(
            lof=lof,
            embedding=modified_embedding[0]
        )

        # ----------------------------------------------------
        # Positive change means:
        #
        # original score > modified score
        #
        # Therefore replacing the feature with its
        # reference value reduced the anomaly score.
        # ----------------------------------------------------

        lof_score_change = (
            original_lof_score
            - modified_lof_score
        )

        # ----------------------------------------------------
        # Absolute effect.
        #
        # Used for ranking influence regardless of direction.
        # ----------------------------------------------------

        absolute_lof_change = abs(
            lof_score_change
        )

        results.append(
            {
                "feature": feature_name,
                "original_value": original_value,
                "reference_value": reference_value,
                "original_lof_score": original_lof_score,
                "lof_score_after_replacement": modified_lof_score,
                "lof_score_change": lof_score_change,
                "absolute_lof_change": absolute_lof_change
            }
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Relative LOF sensitivity percentage.
    #
    # IMPORTANT:
    # This is NOT percentage of anomaly.
    #
    # It is the percentage of the total absolute
    # LOF-score change across the nine individual
    # perturbation experiments.
    # --------------------------------------------------------

    total_absolute_change = (
        results_df[
            "absolute_lof_change"
        ].sum()
    )

    if total_absolute_change > 1e-12:

        results_df[
            "relative_lof_sensitivity_pct"
        ] = (
            results_df[
                "absolute_lof_change"
            ]
            / total_absolute_change
            * 100.0
        )

    else:

        results_df[
            "relative_lof_sensitivity_pct"
        ] = 0.0

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    def get_direction(change):

        if change > 1e-8:

            return "LOF decreased after replacement"

        elif change < -1e-8:

            return "LOF increased after replacement"

        else:

            return "Minimal LOF change"

    results_df[
        "effect_direction"
    ] = results_df[
        "lof_score_change"
    ].apply(
        get_direction
    )

    # --------------------------------------------------------
    # Rank by absolute LOF effect.
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            "absolute_lof_change",
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
        original_lof_score,
        total_absolute_change
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
        / f"{sample_id}_lof_feature_sensitivity.csv"
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
    original_lof_score,
    total_absolute_change
):

    print("\n")

    print("=" * 110)

    print(
        "LOF FEATURE SENSITIVITY ANALYSIS"
    )

    print("=" * 110)

    print(
        f"Sample: {sample_id}"
    )

    print(
        f"Original LOF anomaly score: "
        f"{original_lof_score:.6f}"
    )

    print(
        f"Total absolute LOF change across "
        f"9 perturbations: "
        f"{total_absolute_change:.6f}"
    )

    print("-" * 110)

    print(
        f"{'Rank':<6}"
        f"{'Feature':<28}"
        f"{'Original':>12}"
        f"{'Reference':>12}"
        f"{'LOF Original':>14}"
        f"{'LOF Modified':>14}"
        f"{'LOF Change':>13}"
        f"{'Sensitivity':>14}"
    )

    print("-" * 110)

    for _, row in results_df.iterrows():

        print(
            f"{int(row['rank']):<6}"
            f"{row['feature']:<28}"
            f"{row['original_value']:>12.5f}"
            f"{row['reference_value']:>12.5f}"
            f"{row['original_lof_score']:>14.6f}"
            f"{row['lof_score_after_replacement']:>14.6f}"
            f"{row['lof_score_change']:>13.6f}"
            f"{row['relative_lof_sensitivity_pct']:>13.2f}%"
        )

    print("-" * 110)

    # --------------------------------------------------------
    # Top feature
    # --------------------------------------------------------

    top = results_df.iloc[0]

    print(
        "\nMOST LOF-SENSITIVE FORM FEATURE:"
    )

    print(
        f"  Feature: "
        f"{top['feature']}"
    )

    print(
        f"  Original value: "
        f"{top['original_value']:.6f}"
    )

    print(
        f"  Training reference value: "
        f"{top['reference_value']:.6f}"
    )

    print(
        f"  Original LOF score: "
        f"{top['original_lof_score']:.6f}"
    )

    print(
        f"  LOF after replacement: "
        f"{top['lof_score_after_replacement']:.6f}"
    )

    print(
        f"  LOF score change: "
        f"{top['lof_score_change']:.6f}"
    )

    print(
        f"  Relative LOF sensitivity: "
        f"{top['relative_lof_sensitivity_pct']:.2f}%"
    )

    print(
        f"  Direction: "
        f"{top['effect_direction']}"
    )

    print(
        "\nINTERPRETATION:"
    )

    if top["lof_score_change"] > 0:

        print(
            f"Replacing '{top['feature']}' with its "
            f"training reference value reduced the "
            f"LOF anomaly score by "
            f"{top['lof_score_change']:.6f}."
        )

        print(
            "Therefore, among the nine individual "
            "feature perturbations, this feature had "
            "the largest effect on reducing the "
            "form-level LOF anomaly score."
        )

    elif top["lof_score_change"] < 0:

        print(
            f"Replacing '{top['feature']}' with its "
            f"training reference value increased the "
            f"LOF anomaly score by "
            f"{abs(top['lof_score_change']):.6f}."
        )

        print(
            "Therefore, this feature had the largest "
            "absolute effect on the form-level LOF "
            "score, although the effect increased "
            "rather than reduced the score."
        )

    else:

        print(
            "The individual feature perturbations "
            "produced minimal LOF-score changes."
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The sensitivity percentage represents the "
        "feature's share of the total absolute LOF "
        "change measured across the nine separate "
        "one-feature-at-a-time perturbations."
    )

    print(
        "It is NOT a probability, confidence, severity "
        "percentage, diagnostic measure, or proof that "
        "the feature caused an anomaly."
    )

    print(
        "The analysis also does not capture interactions "
        "between multiple OpenCV features."
    )


# ============================================================
# MAIN
# ============================================================

def normalize_sample_id(raw_id: str) -> str:
    s = str(raw_id).strip()
    if s.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        s = Path(s).stem
    elif "/" in s or "\\" in s:
        s = Path(s).stem
    return s.strip()


def main():

    # --------------------------------------------------------
    # SAMPLE ID
    # --------------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate counterfactual LOF feature sensitivity for a handwriting sample."
    )
    parser.add_argument(
        "pos_sample_id",
        nargs="?",
        default=None,
        help="Sample ID or image filename, e.g. a01-000u or a01-000u.png",
    )
    parser.add_argument(
        "--sample",
        "--sample_id",
        dest="sample_id",
        default=None,
        help="Sample ID or image filename, e.g. a01-000u or a01-000u.png (defaults to a01-000u)",
    )
    args = parser.parse_args()

    raw_sample = args.sample_id or args.pos_sample_id or "a01-000u"
    sample_id = normalize_sample_id(raw_sample)

    print("\n")

    print("=" * 78)

    print(
        "LOF FEATURE SENSITIVITY ANALYSIS"
    )

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
    # Load trained model
    # --------------------------------------------------------

    (
        model,
        feature_mean,
        feature_std
    ) = load_model()

    # --------------------------------------------------------
    # Build LOF reference model
    # --------------------------------------------------------

    (
        lof,
        all_embeddings,
        train_indices,
        validation_indices
    ) = build_lof_reference(
        model=model,
        image_features=image_features,
        opencv_features=opencv_features,
        feature_mean=feature_mean,
        feature_std=feature_std
    )

    # --------------------------------------------------------
    # Find exact sample
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
        f"\nSample index: "
        f"{sample_index}"
    )

    # --------------------------------------------------------
    # Determine whether sample is training or validation.
    # --------------------------------------------------------

    train_index_set = set(
        train_indices.tolist()
    )

    validation_index_set = set(
        validation_indices.tolist()
    )

    if sample_index in train_index_set:

        split_name = "TRAINING"

    elif sample_index in validation_index_set:

        split_name = "VALIDATION"

    else:

        split_name = "UNKNOWN"

    print(
        f"Dataset split: "
        f"{split_name}"
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
    # Print original OpenCV features
    # --------------------------------------------------------

    print(
        "\nOriginal OpenCV features:"
    )

    for name, value in zip(
        FEATURE_NAMES,
        opencv_feature.tolist()
    ):

        print(
            f"  {name:<28}: "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Calculate LOF feature sensitivity
    # --------------------------------------------------------

    (
        results_df,
        original_lof_score,
        total_absolute_change
    ) = calculate_lof_feature_sensitivity(
        model=model,
        lof=lof,
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
        original_lof_score=original_lof_score,
        total_absolute_change=total_absolute_change
    )

    # --------------------------------------------------------
    # Final output
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