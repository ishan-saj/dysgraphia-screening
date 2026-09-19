from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# NOTE ON WHAT THIS SCRIPT IS
#
# Unlike baseline_ocsvm.py and baseline_isolation_forest.py,
# this is a SUPERVISED classifier: it is trained directly on
# the clean-vs-synthetic labels. That makes it a ceiling
# reference, not a fair comparison to the unsupervised k-NN
# pipeline (which never sees a synthetic label during fitting).
# Use it to check whether the corruption is detectable from
# the pixels at all, not as an apples-to-apples baseline.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"

SYNTHETIC_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_form_anomalies.csv"
)

SYNTHETIC_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "images"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "baseline_cnn_results.csv"
)

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 8
RANDOM_STATE = 42


# ============================================================
# DEVICE
# Training a CNN from scratch is much slower than the other
# two baselines, so use a GPU if one is available.
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# BUILD (path, label, anomaly_type) TABLE
# ============================================================

metadata = pd.read_csv(METADATA_FILE)
synthetic = pd.read_csv(SYNTHETIC_FILE)

rows = []

for _, row in metadata.iterrows():

    rows.append({
        "path": str(PROJECT_ROOT / row["image_path"]),
        "label": 0,
        "anomaly_type": "clean"
    })

for _, row in synthetic.iterrows():

    image_path = PROJECT_ROOT / row["synthetic_image"]

    if not image_path.exists():

        image_path = (
            SYNTHETIC_IMAGE_DIR / Path(row["synthetic_image"]).name
        )

    if not image_path.exists():
        continue

    rows.append({
        "path": str(image_path),
        "label": 1,
        "anomaly_type": row["anomaly_type"]
    })

table = pd.DataFrame(rows)

print("Clean images:", (table["label"] == 0).sum())
print("Synthetic images:", (table["label"] == 1).sum())


# ============================================================
# TRAIN / TEST SPLIT
# Stratify on label so both classes are represented in the
# held-out test set that ROC-AUC is computed on.
# ============================================================

train_table, test_table = train_test_split(
    table,
    test_size=0.2,
    stratify=table["label"],
    random_state=RANDOM_STATE
)

print("Train samples:", len(train_table))
print("Test samples:", len(test_table))


# ============================================================
# DATASET
# ============================================================

class FormDataset(Dataset):

    def __init__(self, table):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):

        row = self.table.iloc[index]

        image = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fall back to a blank image rather than crashing
            # a whole epoch on one unreadable file.
            image = np.full(
                (IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8
            )

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        image = torch.from_numpy(image).float() / 255.0

        image = image.unsqueeze(0)  # add channel dimension

        label = torch.tensor(row["label"], dtype=torch.float32)

        return image, label


train_loader = DataLoader(
    FormDataset(train_table),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    FormDataset(test_table),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# SMALL CNN
# Deliberately simple: four conv blocks, one linear head.
# This is a baseline, not a competitor to the fused model.
# ============================================================

class SmallCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        reduced = IMAGE_SIZE // 16  # four 2x2 pooling steps

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced * reduced, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)  # raw logits


model = SmallCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()


# ============================================================
# TRAIN
# ============================================================

print("\nTraining...")

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(train_loader.dataset)

    print(f"Epoch {epoch}/{EPOCHS}  loss = {average_loss:.4f}")


# ============================================================
# EVALUATE ON HELD-OUT TEST SET
# ============================================================

print("\nEvaluating on held-out test set...")

model.eval()

all_scores = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        logits = model(images)

        probabilities = torch.sigmoid(logits).cpu().numpy()

        all_scores.extend(probabilities)
        all_labels.extend(labels.numpy())

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

overall_auc = roc_auc_score(all_labels, all_scores)
overall_ap = average_precision_score(all_labels, all_scores)

print("\n" + "=" * 60)
print("SUPERVISED CNN CLASSIFIER — OVERALL")
print("=" * 60)
print(f"ROC-AUC: {overall_auc:.4f}")
print(f"AP: {overall_ap:.4f}")


# ============================================================
# PER-ANOMALY-TYPE METRICS ON THE TEST SET
# ============================================================

test_table = test_table.reset_index(drop=True)
test_table["score"] = all_scores

clean_test_scores = test_table.loc[
    test_table["label"] == 0, "score"
].values

results = [{
    "anomaly_type": "overall",
    "roc_auc": overall_auc,
    "average_precision": overall_ap
}]

print("\nPer anomaly type (test set only):")

for anomaly_type in sorted(test_table.loc[test_table["label"] == 1, "anomaly_type"].unique()):

    type_scores = test_table.loc[
        (test_table["label"] == 1) & (test_table["anomaly_type"] == anomaly_type),
        "score"
    ].values

    if len(type_scores) == 0:
        continue

    y_type = np.concatenate([
        np.zeros(len(clean_test_scores)),
        np.ones(len(type_scores))
    ])

    scores_type = np.concatenate([clean_test_scores, type_scores])

    auc_type = roc_auc_score(y_type, scores_type)
    ap_type = average_precision_score(y_type, scores_type)

    results.append({
        "anomaly_type": anomaly_type,
        "roc_auc": auc_type,
        "average_precision": ap_type
    })

    print(f"{anomaly_type:12s} ROC-AUC = {auc_type:.4f}  AP = {ap_type:.4f}  (n={len(type_scores)})")


results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved to:")
print(OUTPUT_FILE)