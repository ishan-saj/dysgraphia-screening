from pathlib import Path
import pandas as pd
import torch
from torchvision.models import resnet50, ResNet50_Weights
import cv2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"
)

SYNTHETIC_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "synthetic_form_anomalies.csv"
)

FEATURE_FILE = (
    PROJECT_ROOT / "data" / "metadata" / "resnet_features.pt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
    / "resnet_clean_vs_synthetic.csv"
)

SYNTHETIC_IMAGE_DIR = (
    PROJECT_ROOT
    / "data" / "synthetic_form_anomalies" / "images"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("Device:", device)


# ============================================================
# LOAD DATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)
synthetic = pd.read_csv(SYNTHETIC_FILE)

cached = torch.load(
    FEATURE_FILE,
    map_location="cpu",
    weights_only=False
)

cached_features = cached["image_features"].float()
cached_sample_ids = cached["sample_ids"]


# ============================================================
# FORM ID FIX
# ============================================================

FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}

synthetic["clean_form_id"] = (
    synthetic["source_form"].replace(FORM_ID_MAP)
)


# ============================================================
# BUILD CLEAN LOOKUP
# ============================================================

clean_lookup = {}

for i, row in metadata.iterrows():

    form_id = str(row["form_id"])

    matches = [
        j
        for j, sid in enumerate(cached_sample_ids)
        if str(sid) == str(row["sample_id"])
    ]

    if matches:

        clean_lookup[form_id] = {
            "sample_id": str(row["sample_id"]),
            "feature": cached_features[matches[0]]
        }


print(
    "Clean forms with cached features:",
    len(clean_lookup)
)


# ============================================================
# LOAD RESNET50
# ============================================================

weights = ResNet50_Weights.DEFAULT

resnet = resnet50(weights=weights)

resnet.fc = torch.nn.Identity()

resnet.to(device)
resnet.eval()

print("ResNet50 loaded")


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(path)

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )

    image = torch.from_numpy(
        image
    ).float() / 255.0

    image = image.permute(2, 0, 1)

    return image


# ============================================================
# RESNET FEATURE
# ============================================================

def get_feature(path):

    image = load_image(path)

    image = image.unsqueeze(0)

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    image = (
        image - mean
    ) / std

    with torch.no_grad():

        feature = resnet(
            image.to(device)
        )

    return feature.squeeze(0).cpu()


# ============================================================
# COMPARE
# ============================================================

results = []

print("\nComparing clean vs synthetic ResNet features...")


for index, row in synthetic.iterrows():

    form_id = row["clean_form_id"]

    if form_id not in clean_lookup:
        continue

    clean_feature = clean_lookup[
        form_id
    ]["feature"]


    synthetic_path = (
        PROJECT_ROOT
        / row["synthetic_image"]
    )

    if not synthetic_path.exists():

        synthetic_path = (
            SYNTHETIC_IMAGE_DIR
            / Path(
                row["synthetic_image"]
            ).name
        )


    if not synthetic_path.exists():
        continue


    synthetic_feature = get_feature(
        synthetic_path
    )


    # Euclidean feature difference
    feature_distance = torch.norm(
        clean_feature - synthetic_feature
    ).item()


    # Cosine similarity
    cosine_similarity = torch.nn.functional.cosine_similarity(
        clean_feature.unsqueeze(0),
        synthetic_feature.unsqueeze(0)
    ).item()


    results.append({

        "sample_id":
            row["sample_id"],

        "source_form":
            row["source_form"],

        "anomaly_type":
            row["anomaly_type"],

        "resnet_feature_distance":
            feature_distance,

        "resnet_cosine_similarity":
            cosine_similarity,

    })


    if (index + 1) % 100 == 0:

        print(
            f"Processed {index + 1}/1000"
        )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print("RESNET FEATURE CHANGE")
print("==========================================")

print(
    "Pairs:",
    len(df)
)


print(
    "\nMean ResNet feature distance:"
)

print(
    df.groupby("anomaly_type")[
        "resnet_feature_distance"
    ].agg([
        "count",
        "mean",
        "std",
        "min",
        "max"
    ])
)


print(
    "\nMean cosine similarity:"
)

print(
    df.groupby("anomaly_type")[
        "resnet_cosine_similarity"
    ].mean()
)


print(
    "\nOverall mean feature distance:",
    df["resnet_feature_distance"].mean()
)


print(
    "\nSaved to:"
)

print(OUTPUT_FILE)

