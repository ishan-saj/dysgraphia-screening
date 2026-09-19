import os
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

# ---------------------------------------------------------
# Import the official DTE-C implementation
# ---------------------------------------------------------
DTE_PATH = os.path.expanduser("~/Downloads/DTE-main")
if DTE_PATH not in sys.path:
    sys.path.insert(0, DTE_PATH)

from diffusion.dte import DTECategorical


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_FILE = os.path.join(
    BASE_DIR,
    "data",
    "synthetic_form_anomalies",
    "paired_embeddings.npz"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "synthetic_form_anomalies",
    "baseline_dte_c_results.csv"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
N_SPLITS = 5
SEED = 43

# Start small for the first test.
# The official DTE-C default is 400.
EPOCHS = 400

# Official DTE-C defaults
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 5e-4
T = 400
NUM_BINS = 7


# ---------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------
data = np.load(EMBEDDING_FILE, allow_pickle=True)

clean_embeddings = data["clean_embeddings"]
synthetic_embeddings = data["synthetic_embeddings"]
synthetic_types = data["synthetic_types"]
source_indices = data["source_indices"]

print("Clean embeddings:", clean_embeddings.shape)
print("Synthetic embeddings:", synthetic_embeddings.shape)
print("Source indices:", source_indices.shape)

assert clean_embeddings.shape[1] == 256
assert synthetic_embeddings.shape[1] == 256


# ---------------------------------------------------------
# 5-fold source-aware evaluation
# ---------------------------------------------------------
kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

all_clean_scores = []
all_synthetic_scores = []
all_synthetic_types = []


for fold, (train_idx, val_idx) in enumerate(
    kf.split(clean_embeddings), start=1
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 60)

    # -----------------------------------------------------
    # Training data = clean handwriting only
    # -----------------------------------------------------
    X_train = clean_embeddings[train_idx]

    # -----------------------------------------------------
    # Synthetic samples whose source clean form is in
    # this validation fold
    # -----------------------------------------------------
    synthetic_mask = np.isin(source_indices, val_idx)

    X_synthetic_val = synthetic_embeddings[synthetic_mask]
    types_val = synthetic_types[synthetic_mask]

    # Held-out clean samples
    X_clean_val = clean_embeddings[val_idx]

    print("Training clean:", X_train.shape)
    print("Validation clean:", X_clean_val.shape)
    print("Validation synthetic:", X_synthetic_val.shape)

    # -----------------------------------------------------
    # Standard scaling
    # IMPORTANT:
    # fit scaler ONLY on training clean data
    # -----------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_clean_val_scaled = scaler.transform(X_clean_val)
    X_synthetic_val_scaled = scaler.transform(X_synthetic_val)

    # -----------------------------------------------------
    # Create DTE-C
    # -----------------------------------------------------
    model = DTECategorical(
        seed=SEED,
        hidden_size=[256, 512, 256],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        T=T,
        num_bins=NUM_BINS,
        
    )

    # -----------------------------------------------------
    # Train ONLY on clean training data
    # -----------------------------------------------------
    model.fit(
        X_train_scaled.astype(np.float32),
        verbose=False
    )

    # -----------------------------------------------------
    # Get anomaly scores
    # -----------------------------------------------------
    clean_scores = model.predict_score(
        X_clean_val_scaled.astype(np.float32)
    )

    synthetic_scores = model.predict_score(
        X_synthetic_val_scaled.astype(np.float32)
    )

    # -----------------------------------------------------
    # Store
    # -----------------------------------------------------
    all_clean_scores.extend(clean_scores)
    all_synthetic_scores.extend(synthetic_scores)
    all_synthetic_types.extend(types_val)

    # -----------------------------------------------------
    # Fold ROC-AUC
    # -----------------------------------------------------
    y_fold = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(synthetic_scores))
    ])

    scores_fold = np.concatenate([
        clean_scores,
        synthetic_scores
    ])

    auc = roc_auc_score(y_fold, scores_fold)
    ap = average_precision_score(y_fold, scores_fold)

    print(f"Fold {fold} ROC-AUC: {auc:.4f}")
    print(f"Fold {fold} AP:      {ap:.4f}")


# ---------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------
all_clean_scores = np.asarray(all_clean_scores)
all_synthetic_scores = np.asarray(all_synthetic_scores)
all_synthetic_types = np.asarray(all_synthetic_types)

y = np.concatenate([
    np.zeros(len(all_clean_scores)),
    np.ones(len(all_synthetic_scores))
])

scores = np.concatenate([
    all_clean_scores,
    all_synthetic_scores
])

overall_auc = roc_auc_score(y, scores)
overall_ap = average_precision_score(y, scores)

print("\n" + "=" * 60)
print("OVERALL DTE-C RESULTS")
print("=" * 60)

print(f"Clean samples:      {len(all_clean_scores)}")
print(f"Synthetic samples:  {len(all_synthetic_scores)}")
print(f"ROC-AUC:            {overall_auc:.4f}")
print(f"Average Precision:  {overall_ap:.4f}")


# ---------------------------------------------------------
# Per anomaly type + save result 
# ---------------------------------------------------------
print("\nPER ANOMALY TYPE")
print("-" * 60)

results = []

# Overall result
results.append({
    "anomaly_type": "overall",
    "n": len(all_synthetic_scores),
    "roc_auc": overall_auc,
    "average_precision": overall_ap
})


# Individual anomaly types
for anomaly_type in np.unique(all_synthetic_types):

    mask = all_synthetic_types == anomaly_type

    type_scores = all_synthetic_scores[mask]

    y_type = np.concatenate([
        np.zeros(len(all_clean_scores)),
        np.ones(len(type_scores))
    ])

    scores_type = np.concatenate([
        all_clean_scores,
        type_scores
    ])

    auc = roc_auc_score(y_type, scores_type)
    ap = average_precision_score(y_type, scores_type)

    print(
        f"{anomaly_type:12s} "
        f"n={mask.sum():3d} "
        f"ROC-AUC={auc:.4f} "
        f"AP={ap:.4f}"
    )

    results.append({
        "anomaly_type": anomaly_type,
        "n": int(mask.sum()),
        "roc_auc": auc,
        "average_precision": ap
    })


# ---------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)
print(OUTPUT_FILE)