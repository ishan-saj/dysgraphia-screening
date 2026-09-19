from pathlib import Path
import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import KFold

from model import DysgraphiaModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "metadata_all.csv"
)

SYNTHETIC_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_form_anomalies.csv"
)

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "paired_clean_vs_synthetic.csv"
)

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("Device:", device)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PAIRED CLEAN VS SYNTHETIC ANALYSIS")
print("=" * 70)

metadata = pd.read_csv(METADATA_FILE)
synthetic = pd.read_csv(SYNTHETIC_FILE)

print("\nClean metadata:", len(metadata))
print("Synthetic samples:", len(synthetic))


# ============================================================
# FORM ID NORMALIZATION
# ============================================================

FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}


def normalize_form_id(form_id):

    form_id = str(form_id)

    return FORM_ID_MAP.get(
        form_id,
        form_id
    )


synthetic["clean_form_id"] = (
    synthetic["source_form"]
    .apply(normalize_form_id)
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=device,
    weights_only=False
)

feature_mean = (
    checkpoint["feature_mean"]
    .float()
)

feature_std = (
    checkpoint["feature_std"]
    .float()
)

print("Checkpoint loaded")


# ============================================================
# CREATE MODEL
# ============================================================

model_template = DysgraphiaModel(
    pretrained=False,
    freeze_backbone=True
)


class CachedFusionModel(torch.nn.Module):

    def __init__(self, template):

        super().__init__()

        self.opencv_encoder = (
            template.opencv_encoder
        )

        self.attention = (
            template.attention
        )

        self.projection_head = (
            template.projection_head
        )

        self.opencv_reconstruction = (
            template.opencv_reconstruction
        )

    def forward(
        self,
        image_features,
        opencv_features
    ):

        encoded_opencv = (
            self.opencv_encoder(
                opencv_features
            )
        )

        fused_features = torch.cat(
            [
                image_features,
                encoded_opencv
            ],
            dim=1
        )

        embedding, attention_weights = (
            self.attention(
                fused_features
            )
        )

        projection = (
            self.projection_head(
                embedding
            )
        )

        reconstructed_opencv = (
            self.opencv_reconstruction(
                embedding
            )
        )

        return {
            "embedding": embedding,
            "attention_weights":
                attention_weights,

            "projection":
                projection,

            "reconstructed_opencv":
                reconstructed_opencv
        }


model = CachedFusionModel(
    model_template
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model.to(device)
model.eval()

print("Fusion model loaded")


# ============================================================
# LOAD CACHED CLEAN FEATURES
# ============================================================

print("\nLoading cached clean features...")

cached = torch.load(
    FEATURE_FILE,
    map_location="cpu",
    weights_only=False
)

cached_image_features = (
    cached["image_features"]
    .float()
)

cached_opencv_features = (
    cached["opencv_features"]
    .float()
)

cached_sample_ids = [
    str(x)
    for x in cached["sample_ids"]
]

print(
    "Image features:",
    cached_image_features.shape
)

print(
    "OpenCV features:",
    cached_opencv_features.shape
)


# ============================================================
# CREATE SAMPLE ID LOOKUP
# ============================================================

sample_id_to_index = {}

for i, sample_id in enumerate(
    cached_sample_ids
):

    sample_id_to_index[
        str(sample_id)
    ] = i


# ============================================================
# CREATE FORM → SAMPLE LOOKUP
# ============================================================

form_to_sample = {}

for _, row in metadata.iterrows():

    form_id = str(
        row["form_id"]
    )

    sample_id = str(
        row["sample_id"]
    )

    form_to_sample[
        form_id
    ] = sample_id


# ============================================================
# GENERATE ALL CLEAN EMBEDDINGS
# ============================================================

print(
    "\nGenerating clean embeddings..."
)

with torch.no_grad():

    normalized_opencv = (
        cached_opencv_features
        - feature_mean
    ) / feature_std

    clean_outputs = model(
        cached_image_features,
        normalized_opencv
    )

    all_clean_embeddings = (
        clean_outputs["embedding"]
        .cpu()
    )


print(
    "Clean embeddings:",
    all_clean_embeddings.shape
)


# ============================================================
# VALIDATE SYNTHETIC → CLEAN MATCHING
# ============================================================

print(
    "\nChecking synthetic-to-clean matching..."
)

missing_forms = []

for _, row in synthetic.iterrows():

    form_id = str(
        row["clean_form_id"]
    )

    if form_id not in form_to_sample:

        missing_forms.append(form_id)

if missing_forms:

    print(
        "ERROR: Missing clean forms:"
    )

    print(
        sorted(set(missing_forms))
    )

    raise RuntimeError(
        "Synthetic forms could not "
        "be matched to clean forms."
    )


print(
    "All synthetic forms matched "
    "to clean IAM forms."
)


# ============================================================
# BUILD FORM INDEX
# ============================================================

clean_form_indices = []

clean_form_ids = []

for form_id, sample_id in form_to_sample.items():

    if sample_id not in sample_id_to_index:

        continue

    clean_form_ids.append(
        form_id
    )

    clean_form_indices.append(
        sample_id_to_index[
            sample_id
        ]
    )


clean_form_indices = np.array(
    clean_form_indices
)

clean_form_ids = np.array(
    clean_form_ids
)

print(
    "Forms available for CV:",
    len(clean_form_indices)
)


# ============================================================
# 5-FOLD PAIRED EVALUATION
# ============================================================

print()
print("=" * 70)
print("5-FOLD PAIRED EVALUATION")
print("=" * 70)


kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# This will contain one row per synthetic sample
results = []


for fold_number, (
    train_indices,
    test_indices
) in enumerate(
    kf.split(clean_form_indices),
    start=1
):

    print(
        f"\nFold {fold_number}/{N_SPLITS}"
    )

    # --------------------------------------------------------
    # Training clean forms
    # --------------------------------------------------------

    train_embedding_indices = (
        clean_form_indices[
            train_indices
        ]
    )

    # --------------------------------------------------------
    # Held-out clean forms
    # --------------------------------------------------------

    test_embedding_indices = (
        clean_form_indices[
            test_indices
        ]
    )

    # --------------------------------------------------------
    # Build reference ONLY from training clean forms
    # --------------------------------------------------------

    reference_embedding = (
        all_clean_embeddings[
            train_embedding_indices
        ].mean(dim=0)
    )

    # --------------------------------------------------------
    # Map held-out form IDs
    # --------------------------------------------------------

    held_out_form_ids = set(
        clean_form_ids[
            test_indices
        ]
    )

    # --------------------------------------------------------
    # Process synthetic samples belonging to
    # held-out clean forms
    # --------------------------------------------------------

    fold_results = 0

    for _, row in synthetic.iterrows():

        form_id = str(
            row["clean_form_id"]
        )

        if form_id not in held_out_form_ids:

            continue

        # ----------------------------------------------------
        # Corresponding clean embedding
        # ----------------------------------------------------

        sample_id = form_to_sample[
            form_id
        ]

        clean_index = (
            sample_id_to_index[
                sample_id
            ]
        )

        clean_embedding = (
            all_clean_embeddings[
                clean_index
            ]
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # The synthetic image embedding must come from
        # the paired synthetic dataset.
        #
        # We use the source index saved by
        # save_paired_embeddings.py when available.
        # ----------------------------------------------------

        # Find corresponding synthetic embedding later
        # from paired_embeddings.npz.
        #
        # This script therefore expects that file.
        #
        # We handle that below after loading it.
        fold_results += 1

    print(
        "Held-out synthetic samples:",
        fold_results
    )


# ============================================================
# LOAD PAIRED EMBEDDINGS
# ============================================================

PAIRED_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "paired_embeddings.npz"
)

if not PAIRED_FILE.exists():

    raise FileNotFoundError(
        "\npaired_embeddings.npz not found.\n"
        "Run save_paired_embeddings.py first."
    )


print(
    "\nLoading paired embeddings..."
)

paired = np.load(
    PAIRED_FILE,
    allow_pickle=True
)

paired_clean_embeddings = torch.tensor(
    paired["clean_embeddings"],
    dtype=torch.float32
)

paired_synthetic_embeddings = torch.tensor(
    paired["synthetic_embeddings"],
    dtype=torch.float32
)

paired_anomaly_types = (
    paired["synthetic_types"]
)

source_indices = (
    paired["source_indices"]
)

print(
    "Paired clean embeddings:",
    paired_clean_embeddings.shape
)

print(
    "Paired synthetic embeddings:",
    paired_synthetic_embeddings.shape
)

print(
    "Source indices:",
    source_indices.shape
)


# ============================================================
# SECOND 5-FOLD PASS USING PAIRED EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("CALCULATING PAIRED SCORE CHANGES")
print("=" * 70)


# ------------------------------------------------------------
# Use the unique clean source indices represented by
# the synthetic dataset.
# ------------------------------------------------------------

unique_source_indices = np.unique(
    source_indices
)

print(
    "Unique source forms:",
    len(unique_source_indices)
)


# Map clean feature index → fold assignment
source_to_position = {
    int(source_index): position
    for position, source_index
    in enumerate(unique_source_indices)
}


kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Store score changes
paired_results = []


for fold_number, (
    train_positions,
    test_positions
) in enumerate(
    kf.split(unique_source_indices),
    start=1
):

    print(
        f"Processing fold "
        f"{fold_number}/{N_SPLITS}..."
    )

    train_source_indices = (
        unique_source_indices[
            train_positions
        ]
    )

    test_source_indices = (
        unique_source_indices[
            test_positions
        ]
    )

    # --------------------------------------------------------
    # Reference from TRAINING clean forms only
    # --------------------------------------------------------

    train_mask = np.isin(
        np.arange(
            len(paired_clean_embeddings)
        ),
        train_source_indices
    )

    reference_embedding = (
        paired_clean_embeddings[
            train_source_indices
        ].mean(dim=0)
    )

    # --------------------------------------------------------
    # Evaluate synthetic samples whose source form
    # belongs to held-out set
    # --------------------------------------------------------

    for i in range(
        len(paired_synthetic_embeddings)
    ):

        source_index = int(
            source_indices[i]
        )

        if source_index not in test_source_indices:

            continue

        clean_embedding = (
            paired_clean_embeddings[
                source_index
            ]
        )

        synthetic_embedding = (
            paired_synthetic_embeddings[
                i
            ]
        )

        clean_score = torch.norm(
            clean_embedding
            - reference_embedding
        ).item()

        synthetic_score = torch.norm(
            synthetic_embedding
            - reference_embedding
        ).item()

        score_change = (
            synthetic_score
            - clean_score
        )

        relative_change = (
            score_change
            /
            (clean_score + 1e-8)
        )

        paired_results.append({

            "synthetic_index":
                i,

            "source_index":
                source_index,

            "anomaly_type":
                str(
                    paired_anomaly_types[i]
                ),

            "fold":
                fold_number,

            "clean_score":
                clean_score,

            "synthetic_score":
                synthetic_score,

            "score_change":
                score_change,

            "relative_change":
                relative_change
        })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    paired_results
)

print(
    "\nTotal paired evaluations:",
    len(results_df)
)


# ============================================================
# MERGE ORIGINAL CSV INFORMATION
# ============================================================

if len(results_df) > 0:

    results_df = results_df.merge(
        synthetic[
            [
                "sample_id",
                "source_form",
                "clean_form_id",
                "synthetic_image",
                "ground_truth_mask"
            ]
        ],
        left_on="synthetic_index",
        right_index=True,
        how="left"
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL PAIRED RESULTS")
print("=" * 70)

if len(results_df) > 0:

    print(
        "\nMean score change by anomaly type:"
    )

    summary = (
        results_df
        .groupby(
            "anomaly_type"
        )[
            "score_change"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max"
            ]
        )
    )

    print(summary)

    print(
        "\nMean relative change by anomaly type:"
    )

    relative_summary = (
        results_df
        .groupby(
            "anomaly_type"
        )[
            "relative_change"
        ]
        .mean()
    )

    print(relative_summary)

    print(
        "\nOverall mean clean score:",
        results_df[
            "clean_score"
        ].mean()
    )

    print(
        "Overall mean synthetic score:",
        results_df[
            "synthetic_score"
        ].mean()
    )

    print(
        "Overall mean score change:",
        results_df[
            "score_change"
        ].mean()
    )

else:

    print(
        "ERROR: No paired results were generated."
    )


print(
    "\nResults saved to:"
)

print(OUTPUT_FILE)

print(
    "\nEvaluation complete."
)