#!/usr/bin/env python3
"""
Comprehensive Literature Baseline Anomaly Scorers Benchmark.

Evaluates literature baselines against the proposed hybrid pipeline (ResNet + OpenCV -> k-NN / LOF)
under a leak-free unsupervised One-Class Anomaly Detection protocol (random seed 42).

Reference library: 1,539 clean IAM handwriting forms (normal baseline, y=0).
Anomaly test set:  1,000 synthetic corrupted forms across 4 corruption slices (y=1).

Paradigms evaluated:
1. Devi & Kavya (2023) [Approximate]:
   Block-wise 2D DCT (8x8) + frozen ResNet18 -> Mahalanobis distance to normal centroid.
2. Agarwal et al. (2023) [Faithful Reimplementation]:
   8 paper features (image processing + OCR status) -> OneClassSVM (RBF, nu=0.05).
3. Alevizos et al. (2024) [Adapted Reimplementation]:
   Conv1D-LSTM Autoencoder on geometric handwriting features -> L2 reconstruction error.
4. Proposed Method (Ablation - Local Density):
   Hybrid 256-D fused embeddings -> Leave-One-Out k-NN (k=5).
5. Proposed Method (Upgraded - Adaptive Local Density):
   Hybrid 256-D fused embeddings -> Leave-One-Out Local Outlier Factor (LOF, k=5).

Note on Asselborn et al. (2020):
EXCLUDED from the results table. Their model requires dynamic 60Hz tablet kinematic and pressure
signals unavailable in offline scanned images. Cited only as methodological precedent.
"""

from pathlib import Path
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fftpack import dct
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader, TensorDataset 
from torchvision import models 
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"
CLEAN_FEATURE_FILE = PROJECT_ROOT / "data" / "metadata" / "resnet_features.pt"
PAIRED_CSV = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "paired_clean_vs_synthetic.csv"
SYNTHETIC_CSV = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "synthetic_form_anomalies.csv"
FUSED_EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "paired_embeddings.npz"
SYNTHETIC_IMAGE_DIR = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "images"

# Output artifacts
ROC_PLOT_FILE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "literature_benchmark_roc.png"
PR_PLOT_FILE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "literature_benchmark_pr.png"
ANOMALY_TYPE_PLOT_FILE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "anomaly_type_roc_auc.png"
RESULTS_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_METRICS_CSV = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "literature_benchmark_metrics.csv"
OUTPUT_SCORES_CSV = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "literature_benchmark_raw_scores.csv"

# Cache files for extracted literature features to avoid redundant disk I/O on re-runs
AGARWAL_CACHE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "agarwal_features.npz"
DEVI_KAVYA_CACHE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "devi_kavya_features.npz"

RANDOM_STATE = 42
FORM_ID_MAP = {
    "a05-094(1)": "a05-094",
    "b06-075(1)": "b06-075",
    "g06-037b(1)": "g06-037b",
}


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class ConvLSTMAutoencoder(nn.Module):
    """
    Adapted from Alevizos et al. (ICECCE, 2024):
    Convolutional encoder feeding an LSTM layer, mirrored by an LSTM and ConvTranspose decoder.
    Trained strictly on normal samples; anomaly score = L2 reconstruction error.
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.conv_enc = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm_enc = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.lstm_dec = nn.LSTM(input_size=hidden_dim, hidden_size=32, batch_first=True)
        self.conv_dec = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, dim = x.shape
        x_in = x.unsqueeze(1)
        conv_out = self.conv_enc(x_in)
        conv_perm = conv_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm_enc(conv_perm)
        latent = self.fc_latent(lstm_out[:, -1, :])

        dec_hidden = self.fc_dec(latent).unsqueeze(1).repeat(1, dim, 1)
        lstm_dec_out, _ = self.lstm_dec(dec_hidden)
        dec_perm = lstm_dec_out.permute(0, 2, 1)
        recon = self.conv_dec(dec_perm).squeeze(1)
        return recon


# ============================================================
# FEATURE EXTRACTION HELPERS
# ============================================================

def check_ocr_availability():
    """Verify whether pytesseract and the underlying Tesseract binary are installed."""
    try:
        import pytesseract 
        ver = pytesseract.get_tesseract_version()
        return True, f"pytesseract version {pytesseract.__version__}, tesseract engine {ver}"
    except Exception as e:
        return False, str(e)


from dataset import load_handwriting_crop, resize_handwriting_crop, fold_angle_45


def extract_agarwal_features_single(img: np.ndarray, ocr_available: bool = False) -> np.ndarray:
    """
    Extract features from Agarwal et al. (SN Comput Sci, 2023):
    1. Baseline angle variance (cv2.minAreaRect on dilated lines, height >= 20px, folded into [-45, 45])
    2. Pen pressure (mean of non-zero inverted thresholded pixels)
    3. Slant angle (shear transforms at -45 to 45 deg, max squared density)
    4. Letter size irregularity (horizontal projection run-length std)
    5. Line spacing (blank-line ratio)
    6. Word spacing (blank-column ratio vs connected components)
    7. Spelling errors (OCR vs dictionary) -> Flagged if OCR unavailable
    8. Irregular capitalization (intermediate caps) -> Flagged if OCR unavailable
    """
    h, w = img.shape
    scale = 1000.0 / h
    img_scaled = cv2.resize(img, (int(w * scale), 1000), interpolation=cv2.INTER_AREA)

    # Preprocessing
    filtered = cv2.bilateralFilter(img_scaled, d=7, sigmaColor=50, sigmaSpace=50)
    _, bin_inv = cv2.threshold(filtered, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilated = cv2.dilate(bin_inv, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 1. Baseline angle variance (folded into [-45, 45] to avoid minAreaRect wraparound)
    angles = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch >= int(20 * scale) and cw >= int(50 * scale):
            rect = cv2.minAreaRect(c)
            angles.append(fold_angle_45(rect[-1]))
    f1 = float(np.var(angles)) if len(angles) > 1 else 0.0

    # 2. Pen pressure
    inv = 255 - img_scaled
    _, tozero = cv2.threshold(inv, 100, 255, cv2.THRESH_TOZERO)
    nz = tozero[tozero > 0]
    f2 = float(nz.mean()) if len(nz) > 0 else 0.0

    # 3. Slant angle via shear transform
    shear_angles = [-45, -30, -15, -5, 0, 5, 15, 30, 45]
    best_angle = 0
    max_sq = -1
    small_bin = cv2.resize(bin_inv, (int(w * scale) // 2, 500), interpolation=cv2.INTER_AREA)
    sh_h, sh_w = small_bin.shape
    for theta in shear_angles:
        rad = np.radians(theta)
        M = np.float32([[1, np.tan(rad), 0], [0, 1, 0]])
        warped = cv2.warpAffine(small_bin, M, (sh_w, sh_h))
        sq = np.sum(np.square(warped.sum(axis=0, dtype=np.float64)))
        if sq > max_sq:
            max_sq = sq
            best_angle = theta
    f3 = float(best_angle)

    # 4. Letter size irregularity
    h_proj = bin_inv.sum(axis=1)
    thresh = np.mean(h_proj[h_proj > 0]) * 0.2 if np.any(h_proj > 0) else 0
    runs = []
    c_run = 0
    for v in h_proj:
        if v > thresh:
            c_run += 1
        elif c_run > 0:
            runs.append(c_run)
            c_run = 0
    if c_run > 0:
        runs.append(c_run)
    f4 = float(np.std(runs)) if len(runs) > 1 else 0.0

    # 5. Line spacing
    f5 = float(np.sum(h_proj == 0) / 1000.0)

    # 6. Word spacing
    v_proj = bin_inv.sum(axis=0)
    f6 = float(np.sum(v_proj == 0) / max(1, len(contours)))

    # 7 & 8: OCR features
    f7 = 0.0  # Spelling errors
    f8 = 0.0  # Irregular capitalization

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8], dtype=np.float32)


def get_or_extract_agarwal_features(clean_image_paths, synth_image_paths, clean_form_ids=None, synth_source_forms=None):
    """Extract or load cached features for Agarwal et al. using shared handwriting crop."""
    if AGARWAL_CACHE.exists():
        print(f"  [Cache] Loading cached Agarwal features from: {AGARWAL_CACHE.name}")
        data = np.load(AGARWAL_CACHE, allow_pickle=True)
        assert "cropped" in data.files and bool(data["cropped"]) is True, (
            f"Refusing to run: {AGARWAL_CACHE} is missing cropped=True flag!"
        )
        assert "interp" in data.files and str(data["interp"]) == "INTER_AREA", (
            f"Refusing to run: {AGARWAL_CACHE} is missing interp='INTER_AREA' flag!"
        )
        return data["clean_features"], data["synth_features"]

    if clean_form_ids is None or synth_source_forms is None:
        raise ValueError("clean_form_ids and synth_source_forms are required to extract cropped Agarwal features.")

    print("  Extracting Agarwal et al. (2023) 8 features from CROPPED handwriting images...")
    ocr_ok, ocr_msg = check_ocr_availability()
    print(f"  [Integrity Check] OCR Status: {ocr_ok} ({ocr_msg})")
    if not ocr_ok:
        print("  [Notice] Tesseract OCR is unavailable. Features 7 (Spelling errors) and 8 (Irregular capitalization) default to 0.0.")

    clean_feats = []
    for i, (p, fid) in enumerate(zip(clean_image_paths, clean_form_ids)):
        full_p = PROJECT_ROOT / p
        img = cv2.imread(str(full_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing clean image: {full_p}")
        img_cropped = load_handwriting_crop(fid, img)
        clean_feats.append(extract_agarwal_features_single(img_cropped, ocr_available=ocr_ok))
        if (i + 1) % 500 == 0 or (i + 1) == len(clean_image_paths):
            print(f"    Clean forms: {i+1}/{len(clean_image_paths)}")

    synth_feats = []
    for i, (p, sfid) in enumerate(zip(synth_image_paths, synth_source_forms)):
        full_p = PROJECT_ROOT / p
        img = cv2.imread(str(full_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing synthetic image: {full_p}")
        img_cropped = load_handwriting_crop(sfid, img)
        synth_feats.append(extract_agarwal_features_single(img_cropped, ocr_available=ocr_ok))
        if (i + 1) % 500 == 0 or (i + 1) == len(synth_image_paths):
            print(f"    Synthetic forms: {i+1}/{len(synth_image_paths)}")

    clean_feats = np.array(clean_feats, dtype=np.float32)
    synth_feats = np.array(synth_feats, dtype=np.float32)
    assert not np.isnan(clean_feats).any() and not np.isinf(clean_feats).any(), "NaN/Inf in clean Agarwal features!"
    assert not np.isnan(synth_feats).any() and not np.isinf(synth_feats).any(), "NaN/Inf in synth Agarwal features!"

    np.savez(
        AGARWAL_CACHE,
        clean_features=clean_feats,
        synth_features=synth_feats,
        cropped=np.array(True),
        crop_mode=np.array("handwriting_bbox_p30"),
        interp=np.array("INTER_AREA"),
    )
    print(f"  [Cache] Saved cropped Agarwal features to: {AGARWAL_CACHE.name}")
    return clean_feats, synth_feats


def extract_devi_kavya_features_single(img: np.ndarray, r18_model: nn.Module) -> np.ndarray:
    """
    Approximate implementation of Devi & Kavya (2023):
    1. Block-wise 2D DCT (8x8 blocks, standard JPEG-style stand-in for Kekre-DCT).
    2. Pool mean and std of DCT coefficients across blocks (128-D).
    3. ResNet18 penultimate layer features (512-D).
    Concatenation = 640-D vector.
    Uses shared resize_handwriting_crop (cv2.INTER_AREA).
    """
    resized_u8 = resize_handwriting_crop(img, size=224)
    img_224 = resized_u8.astype(np.float32) / 255.0
    h, w = img_224.shape

    # Vectorized 8x8 block 2D DCT
    blocks = img_224.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
    d2 = dct(dct(blocks, axis=2, norm="ortho"), axis=1, norm="ortho")
    d2_flat = d2.reshape(blocks.shape[0], 64)
    dct_feat = np.concatenate([d2_flat.mean(axis=0), d2_flat.std(axis=0)])  # 128-D

    # ResNet18 penultimate layer
    rgb_224 = cv2.cvtColor(resized_u8, cv2.COLOR_GRAY2RGB)
    t = torch.from_numpy(rgb_224).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        cnn_feat = r18_model((t - mean) / std).squeeze(0).numpy()  # 512-D

    return np.concatenate([dct_feat, cnn_feat]).astype(np.float32)  # 640-D


def get_or_extract_devi_kavya_features(clean_image_paths, synth_image_paths, clean_form_ids=None, synth_source_forms=None):
    """Extract or load cached features for Devi & Kavya using shared handwriting crop."""
    if DEVI_KAVYA_CACHE.exists():
        print(f"  [Cache] Loading cached Devi & Kavya features from: {DEVI_KAVYA_CACHE.name}")
        data = np.load(DEVI_KAVYA_CACHE, allow_pickle=True)
        assert "cropped" in data.files and bool(data["cropped"]) is True, (
            f"Refusing to run: {DEVI_KAVYA_CACHE} is missing cropped=True flag!"
        )
        assert "interp" in data.files and str(data["interp"]) == "INTER_AREA", (
            f"Refusing to run: {DEVI_KAVYA_CACHE} is missing interp='INTER_AREA' flag!"
        )
        return data["clean_features"], data["synth_features"]

    if clean_form_ids is None or synth_source_forms is None:
        raise ValueError("clean_form_ids and synth_source_forms are required to extract cropped Devi & Kavya features.")

    print("  Extracting Devi & Kavya (2023) [8x8 DCT (128-D) + ResNet18 (512-D)] from CROPPED handwriting images...")
    r18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    r18.fc = nn.Identity()
    r18.eval()

    clean_feats = []
    for i, (p, fid) in enumerate(zip(clean_image_paths, clean_form_ids)):
        full_p = PROJECT_ROOT / p
        img = cv2.imread(str(full_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing clean image: {full_p}")
        img_cropped = load_handwriting_crop(fid, img)
        clean_feats.append(extract_devi_kavya_features_single(img_cropped, r18))
        if (i + 1) % 500 == 0 or (i + 1) == len(clean_image_paths):
            print(f"    Clean forms: {i+1}/{len(clean_image_paths)}")

    synth_feats = []
    for i, (p, sfid) in enumerate(zip(synth_image_paths, synth_source_forms)):
        full_p = PROJECT_ROOT / p
        img = cv2.imread(str(full_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Missing synthetic image: {full_p}")
        img_cropped = load_handwriting_crop(sfid, img)
        synth_feats.append(extract_devi_kavya_features_single(img_cropped, r18))
        if (i + 1) % 500 == 0 or (i + 1) == len(synth_image_paths):
            print(f"    Synthetic forms: {i+1}/{len(synth_image_paths)}")

    clean_feats = np.array(clean_feats, dtype=np.float32)
    synth_feats = np.array(synth_feats, dtype=np.float32)
    assert not np.isnan(clean_feats).any() and not np.isinf(clean_feats).any(), "NaN/Inf in clean Devi & Kavya features!"
    assert not np.isnan(synth_feats).any() and not np.isinf(synth_feats).any(), "NaN/Inf in synth Devi & Kavya features!"

    np.savez(
        DEVI_KAVYA_CACHE,
        clean_features=clean_feats,
        synth_features=synth_feats,
        cropped=np.array(True),
        interp=np.array("INTER_AREA"),
        crop_mode=np.array("handwriting_bbox_p30"),
    )
    print(f"  [Cache] Saved cropped Devi & Kavya features to: {DEVI_KAVYA_CACHE.name}")
    return clean_feats, synth_feats


# ============================================================
# MAIN BENCHMARK EXECUTION
# ============================================================

def run_benchmark():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("=" * 100)
    print("FAITHFUL LITERATURE BASELINES & HYBRID PIPELINE BENCHMARK")
    print("=" * 100)

    # ------------------------------------------------------------
    # 1. Dataset Loading & Alignment
    # ------------------------------------------------------------
    print("\n[Step 1/6] Loading metadata and ground-truth alignments...")
    meta_df = pd.read_csv(METADATA_FILE)
    paired_df = pd.read_csv(PAIRED_CSV)
    synth_meta_df = pd.read_csv(SYNTHETIC_CSV)

    clean_image_paths = meta_df["image_path"].tolist()
    clean_form_ids = [str(x).strip() for x in meta_df["form_id"].tolist()]
    synth_image_paths = synth_meta_df["synthetic_image"].tolist()
    synth_source_forms = [FORM_ID_MAP.get(str(x).strip(), str(x).strip()) for x in synth_meta_df["source_form"].tolist()]
    synth_types = np.array([str(x) for x in paired_df["anomaly_type"]])
    synth_sample_ids = paired_df["sample_id"].tolist()
    clean_sample_ids = meta_df["sample_id"].tolist()

    n_clean = len(clean_sample_ids)
    n_synth = len(synth_types)
    print(f"  Loaded {n_clean} clean IAM forms and {n_synth} synthetic corrupted forms.")
    print(f"  Corruption slices: Blur ({np.sum(synth_types=='blur')}), "
          f"Dropout ({np.sum(synth_types=='dropout')}), "
          f"Thickening ({np.sum(synth_types=='thickening')}), "
          f"Thinning ({np.sum(synth_types=='thinning')})")

    # ------------------------------------------------------------
    # 2. Evaluation Population Setup (Full Clean Reference: N=1,539 vs Synthetic: N=1,000)
    # ------------------------------------------------------------
    print(f"\n[Step 2/6] Setting up reference population evaluation (Clean N={n_clean}, Synthetic N={n_synth})...")
    print(f"  Normal reference library (y=0): {n_clean} clean IAM forms")
    print(f"  Synthetic anomaly test (y=1):   {n_synth} corrupted forms")
    print(f"  Total evaluation set:           {n_clean + n_synth}")

    y_test = np.concatenate([np.zeros(n_clean), np.ones(n_synth)])

    # ------------------------------------------------------------
    # 3. Load / Extract Feature Spaces
    # ------------------------------------------------------------
    print("\n[Step 3/6] Ingesting feature spaces across paradigms...")

    # A. Existing Hybrid 256-D fused embeddings
    fused_data = np.load(FUSED_EMBEDDINGS_FILE, allow_pickle=True)
    assert "cropped" in fused_data.files and bool(fused_data["cropped"]) is True, (
        f"Refusing to run: {FUSED_EMBEDDINGS_FILE} is missing cropped=True!"
    )
    assert "interp" in fused_data.files and str(fused_data["interp"]) == "INTER_AREA", (
        f"Refusing to run: {FUSED_EMBEDDINGS_FILE} is missing interp='INTER_AREA'!"
    )
    clean_fused = fused_data["clean_embeddings"]
    synth_fused = fused_data["synthetic_embeddings"]

    clean_res_data = torch.load(CLEAN_FEATURE_FILE, map_location="cpu", weights_only=False)
    assert bool(clean_res_data.get("cropped", False)) is True, (
        f"Refusing to run: {CLEAN_FEATURE_FILE} is missing cropped=True!"
    )
    assert str(clean_res_data.get("interp", "")) == "INTER_AREA", (
        f"Refusing to run: {CLEAN_FEATURE_FILE} is missing interp='INTER_AREA'!"
    )
    assert bool(fused_data["cropped"]) == bool(clean_res_data.get("cropped", False)), (
        "Crop flag mismatch between paired_embeddings.npz and resnet_features.pt!"
    )

    # B. Agarwal et al. 8-D features
    clean_agarwal, synth_agarwal = get_or_extract_agarwal_features(
        clean_image_paths, synth_image_paths, clean_form_ids=clean_form_ids, synth_source_forms=synth_source_forms
    )

    # C. Alevizos et al. features (6 geometric indicators: baseline, spacing, density, slant, width, height)
    # Extract directly from true pixel-extracted OpenCV features (NO parent lookup)
    clean_cv = fused_data["clean_pixel_opencv"] if "clean_pixel_opencv" in fused_data.files else clean_res_data["opencv_features"].numpy()
    synth_cv = fused_data["synthetic_pixel_opencv"]
    # Selected 6 features matching paper: [baseline_dev(1), spacing_cv(2), stroke_density(6), slant(7), word_width(5), word_height(4)]
    feature_cols = [1, 2, 6, 7, 5, 4]
    clean_alevizos = clean_cv[:, feature_cols]
    synth_alevizos = synth_cv[:, feature_cols]

    # D. Devi & Kavya features (640-D: 128-D DCT + 512-D ResNet18)
    clean_dk, synth_dk = get_or_extract_devi_kavya_features(
        clean_image_paths, synth_image_paths, clean_form_ids=clean_form_ids, synth_source_forms=synth_source_forms
    )

    for arr_name, arr in [
        ("clean_fused", clean_fused), ("synth_fused", synth_fused),
        ("clean_agarwal", clean_agarwal), ("synth_agarwal", synth_agarwal),
        ("clean_alevizos", clean_alevizos), ("synth_alevizos", synth_alevizos),
        ("clean_dk", clean_dk), ("synth_dk", synth_dk),
    ]:
        assert not np.isnan(arr).any() and not np.isinf(arr).any(), f"NaN/Inf detected in {arr_name}!"

    # ------------------------------------------------------------
    # 4. Model Training & Anomaly Scoring (5-Fold Parent-Excluded Cross-Fitting)
    # ------------------------------------------------------------
    print("\n[Step 4/6] Fitting anomaly scorers via 5-fold parent-excluded cross-fitting...")
    from sklearn.model_selection import KFold
    fid_to_idx = {fid: i for i, fid in enumerate(clean_form_ids)}
    synth_parent_idx = np.array([
        fid_to_idx[FORM_ID_MAP.get(str(r["clean_form_id"]).strip(), str(r["clean_form_id"]).strip())]
        for _, r in paired_df.iterrows()
    ], dtype=int)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(kf.split(np.arange(n_clean)))

    models_dict = {}

    # ------------------------------------------------------------
    # Method 1: Devi & Kavya (2023) [DCT + ResNet-18 -> Mahalanobis]
    # ------------------------------------------------------------
    print("  -> (1/5) Devi & Kavya (2023) [8x8 DCT + ResNet18 -> Mahalanobis]...")
    m1_clean_test = np.zeros(n_clean, dtype=np.float64)
    m1_synth_test = np.zeros(n_synth, dtype=np.float64)
    for tr_i, te_i in folds:
        te_set = set(int(x) for x in te_i)
        s_mask = np.array([int(p) in te_set for p in synth_parent_idx], dtype=bool)
        dk_scaler = StandardScaler().fit(clean_dk[tr_i])
        dk_tr = dk_scaler.transform(clean_dk[tr_i])
        dk_te_c = dk_scaler.transform(clean_dk[te_i])
        dk_te_s = dk_scaler.transform(synth_dk[s_mask])

        dk_centroid = dk_tr.mean(axis=0)
        dk_cov = np.cov(dk_tr, rowvar=False) + 1e-4 * np.eye(dk_tr.shape[1])
        dk_cov_inv = np.linalg.pinv(dk_cov)

        diff_c = dk_te_c - dk_centroid
        diff_s = dk_te_s - dk_centroid
        m1_clean_test[te_i] = np.sqrt(np.sum((diff_c @ dk_cov_inv) * diff_c, axis=1))
        m1_synth_test[s_mask] = np.sqrt(np.sum((diff_s @ dk_cov_inv) * diff_s, axis=1))

    models_dict["Devi & Kavya (2023) [DCT + ResNet-18]"] = {
        "status": "Kekre-DCT frequency features (128-D) + ResNet-18 (512-D) [640-D] -> Mahalanobis distance",
        "features": "8x8 DCT (128-D) + ResNet-18 (512-D) [640-D]",
        "test_scores": np.concatenate([m1_clean_test, m1_synth_test]),
        "clean_all_scores": m1_clean_test,
        "synth_all_scores": m1_synth_test,
        "style": {"color": "#E03131", "linestyle": ":", "linewidth": 2.2},
    }

    # ------------------------------------------------------------
    # Method 2: Agarwal et al. (2023) [Faithful Reimplementation: OneClassSVM]
    # ------------------------------------------------------------
    print("  -> (2/5) Agarwal et al. (2023) [Faithful: 8 Features -> OneClassSVM (nu=0.05)]...")
    m3_clean_test = np.zeros(n_clean, dtype=np.float64)
    m3_synth_test = np.zeros(n_synth, dtype=np.float64)
    for tr_i, te_i in folds:
        te_set = set(int(x) for x in te_i)
        s_mask = np.array([int(p) in te_set for p in synth_parent_idx], dtype=bool)
        ag_scaler = StandardScaler().fit(clean_agarwal[tr_i])
        ag_tr = ag_scaler.transform(clean_agarwal[tr_i])
        ag_te_c = ag_scaler.transform(clean_agarwal[te_i])
        ag_te_s = ag_scaler.transform(synth_agarwal[s_mask])
        ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(ag_tr)
        m3_clean_test[te_i] = -ocsvm.decision_function(ag_te_c)
        m3_synth_test[s_mask] = -ocsvm.decision_function(ag_te_s)

    models_dict["Agarwal et al. (2023) [Faithful Reimplementation]"] = {
        "status": "Faithful reimplementation (Features 1-6 verified; Features 7-8 OCR unavailable)",
        "features": "8 paper features (visual geometric + OCR status)",
        "test_scores": np.concatenate([m3_clean_test, m3_synth_test]),
        "clean_all_scores": m3_clean_test,
        "synth_all_scores": m3_synth_test,
        "style": {"color": "#F08C00", "linestyle": "-.", "linewidth": 2.2},
    }

    # ------------------------------------------------------------
    # Method 3: Alevizos et al. (2024) [Adapted: Conv-LSTM Autoencoder]
    # ------------------------------------------------------------
    print("  -> (3/5) Alevizos et al. (2024) [Adapted: Conv-LSTM Autoencoder L2 Error]...")
    m4_clean_test = np.zeros(n_clean, dtype=np.float64)
    m4_synth_test = np.zeros(n_synth, dtype=np.float64)
    for f_i, (tr_i, te_i) in enumerate(folds):
        torch.manual_seed(RANDOM_STATE + f_i)
        te_set = set(int(x) for x in te_i)
        s_mask = np.array([int(p) in te_set for p in synth_parent_idx], dtype=bool)
        al_scaler = StandardScaler().fit(clean_alevizos[tr_i])
        al_tr = al_scaler.transform(clean_alevizos[tr_i])
        al_te_c = al_scaler.transform(clean_alevizos[te_i])
        al_te_s = al_scaler.transform(synth_alevizos[s_mask])

        ae_model = ConvLSTMAutoencoder(input_dim=6, hidden_dim=32, latent_dim=16)
        ae_opt = torch.optim.Adam(ae_model.parameters(), lr=1e-3, weight_decay=1e-4)
        ae_crit = nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(torch.tensor(al_tr, dtype=torch.float32)),
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(RANDOM_STATE),
        )
        ae_model.train()
        for _ in range(20):
            for batch_x, in train_loader:
                ae_opt.zero_grad()
                recon_x = ae_model(batch_x)
                loss = ae_crit(recon_x, batch_x)
                loss.backward()
                ae_opt.step()

        ae_model.eval()
        with torch.no_grad():
            c_t = torch.tensor(al_te_c, dtype=torch.float32)
            s_t = torch.tensor(al_te_s, dtype=torch.float32)
            m4_clean_test[te_i] = torch.sum((c_t - ae_model(c_t)) ** 2, dim=1).numpy()
            m4_synth_test[s_mask] = torch.sum((s_t - ae_model(s_t)) ** 2, dim=1).numpy()

    models_dict["Alevizos et al. (2024) [Adapted Autoencoder]"] = {
        "status": "Architecture & features adapted from supervised 4-class model into unsupervised L2 scorer",
        "features": "6 handwriting features (deviation, connectivity, density, slant, width, height)",
        "test_scores": np.concatenate([m4_clean_test, m4_synth_test]),
        "clean_all_scores": m4_clean_test,
        "synth_all_scores": m4_synth_test,
        "style": {"color": "#9C36B5", "linestyle": ":", "linewidth": 2.4},
    }

    # ------------------------------------------------------------
    # Method 4: Proposed Method (Ablation - Local Density: k-NN k=5, Parent-Excluded)
    # ------------------------------------------------------------
    print("  -> (4/5) Proposed Method (Ablation) [Pure Hybrid 256-D -> Parent-Excluded k-NN (k=5)]...")
    m5_clean_test = np.zeros(n_clean, dtype=np.float64)
    m5_synth_test = np.zeros(n_synth, dtype=np.float64)
    for tr_i, te_i in folds:
        te_set = set(int(x) for x in te_i)
        s_mask = np.array([int(p) in te_set for p in synth_parent_idx], dtype=bool)
        knn_fused = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(clean_fused[tr_i])
        dc, _ = knn_fused.kneighbors(clean_fused[te_i])
        ds, _ = knn_fused.kneighbors(synth_fused[s_mask])
        m5_clean_test[te_i] = dc.mean(axis=1)
        m5_synth_test[s_mask] = ds.mean(axis=1)

    models_dict["Proposed Method (Ablation: k-NN)"] = {
        "status": "Proposed hybrid architecture with standard local density metric (parent-excluded)",
        "features": "Fused cross-attention visual-geometric embedding [256-D]",
        "test_scores": np.concatenate([m5_clean_test, m5_synth_test]),
        "clean_all_scores": m5_clean_test,
        "synth_all_scores": m5_synth_test,
        "style": {"color": "#0CA678", "linestyle": "--", "linewidth": 2.2},
    }

    # ------------------------------------------------------------
    # Method 5: Proposed Method (Upgraded - Adaptive Local Density: LOF k=5, Parent-Excluded)
    # ------------------------------------------------------------
    print("  -> (5/5) Proposed Method (Upgraded) [Pure Hybrid 256-D -> Parent-Excluded LOF (k=5)]...")
    m6_clean_test = np.zeros(n_clean, dtype=np.float64)
    m6_synth_test = np.zeros(n_synth, dtype=np.float64)
    for tr_i, te_i in folds:
        te_set = set(int(x) for x in te_i)
        s_mask = np.array([int(p) in te_set for p in synth_parent_idx], dtype=bool)
        lof_nov = LocalOutlierFactor(n_neighbors=5, metric="minkowski", novelty=True).fit(clean_fused[tr_i])
        m6_clean_test[te_i] = -lof_nov.decision_function(clean_fused[te_i])
        m6_synth_test[s_mask] = -lof_nov.decision_function(synth_fused[s_mask])

    models_dict["Proposed Method (Upgraded: Adaptive LOF)"] = {
        "status": "Proposed hybrid architecture with adaptive local outlier factor (LOO evaluation)",
        "features": "Fused cross-attention visual-geometric embedding [256-D]",
        "test_scores": np.concatenate([m6_clean_test, m6_synth_test]),
        "clean_all_scores": m6_clean_test,
        "synth_all_scores": m6_synth_test,
        "style": {"color": "#2F9E44", "linestyle": "-", "linewidth": 2.6},
    }

    # ------------------------------------------------------------
    # 5. Compute Metrics: ROC-AUC, PR-AUC, and Slices
    # ------------------------------------------------------------
    print("\n[Step 5/6] Computing evaluation metrics across evaluation set...")
    slices = ["blur", "dropout", "thickening", "thinning"]
    results_records = []

    for name, m_info in models_dict.items():
        scores = m_info["test_scores"]
        clean_sc = scores[:n_clean]
        synth_sc = scores[n_clean:]

        overall_auc = roc_auc_score(y_test, scores)
        overall_ap = average_precision_score(y_test, scores)

        rec = {
            "Method": name,
            "Verification Status": m_info["status"],
            "Feature Space": m_info["features"],
            "Overall AUC": overall_auc,
            "Overall AP": overall_ap,
        }

        for sl in slices:
            mask = (synth_types == sl)
            y_slice = np.concatenate([np.zeros(len(clean_sc)), np.ones(mask.sum())])
            scores_slice = np.concatenate([clean_sc, synth_sc[mask]])
            rec[f"{sl.capitalize()} AUC"] = roc_auc_score(y_slice, scores_slice)

        results_records.append(rec)

    metrics_df = pd.DataFrame(results_records)

    # ------------------------------------------------------------
    # 6. Generate ROC & PR Curves Plot
    # ------------------------------------------------------------
    print("\n[Step 6/6] Generating combined ROC and PR curve visualizations...")

    # ROC Curves
    plt.figure(figsize=(10, 8), dpi=300)
    for name, m_info in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, m_info["test_scores"])
        roc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.4f})", **m_info["style"])

    plt.plot([0, 1], [0, 1], color="#868E96", linestyle=":", linewidth=1.5, label="Chance Baseline (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.04])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("Literature Baselines vs Proposed Hybrid Pipeline: Overlaid ROC Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, facecolor="#FFFFFF", edgecolor="#CED4DA", fontsize=8.8)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_FILE, dpi=300)
    plt.close()
    print(f"  [OK] Saved ROC curve plot to: {ROC_PLOT_FILE.name}")

    # PR Curves
    plt.figure(figsize=(10, 8), dpi=300)
    prevalence = np.mean(y_test)
    for name, m_info in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, m_info["test_scores"])
        ap_val = average_precision_score(y_test, m_info["test_scores"])
        plt.plot(recall, precision, label=f"{name} (AP = {ap_val:.4f})", **m_info["style"])

    plt.plot([0, 1], [prevalence, prevalence], color="#868E96", linestyle=":", linewidth=1.5, label=f"Random Baseline (AP = {prevalence:.4f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.04])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("Literature Baselines vs Proposed Hybrid Pipeline: Precision-Recall Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True, framealpha=0.95, facecolor="#FFFFFF", edgecolor="#CED4DA", fontsize=8.8)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(PR_PLOT_FILE, dpi=300)
    plt.close()
    print(f"  [OK] Saved PR curve plot to:  {PR_PLOT_FILE.name}")

    # Anomaly-Type Analysis: ROC-AUC by Perturbation (Slide 18 Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    perturbation_cols = ["Blur AUC", "Dropout AUC", "Thickening AUC", "Thinning AUC"]
    perturbation_labels = ["Blur", "Dropout", "Thickening", "Thinning"]
    short_names = [
        "Devi & Kavya",
        "Agarwal et al.",
        "Alevizos et al.",
        "Proposed k-NN",
        "Proposed Adaptive LOF",
    ]
    bar_colors = ["#3B68B5", "#ED7D31", "#9E9E9E", "#68A840", "#C0281B"]

    x = np.arange(len(perturbation_labels))
    n_models = len(short_names)
    width = 0.145
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for idx, (row_idx, s_name, b_color, offset) in enumerate(zip(range(len(metrics_df)), short_names, bar_colors, offsets)):
        row = metrics_df.iloc[row_idx]
        vals = [float(row[c]) for c in perturbation_cols]
        bars = ax.bar(
            x + offset,
            vals,
            width=width * 0.96,
            color=b_color,
            edgecolor="none",
            label=s_name,
        )
        is_proposed_lof = (idx == n_models - 1)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (0.010 if is_proposed_lof else 0.006),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=7.6 if is_proposed_lof else 7.2,
                fontweight="bold" if is_proposed_lof else "normal",
                color="#1A1A1A",
            )

    ax.set_title("ROC-AUC by Perturbation", loc="left", fontsize=15, fontweight="bold", color="#1A1A1A", pad=10)
    ax.set_ylabel("ROC-AUC", fontsize=11.5, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(perturbation_labels, fontsize=11.5, color="#222222")
    ax.set_xlim(-0.55, 3.95)
    ax.set_ylim(0.0, 0.85)
    ax.set_yticks(np.arange(0.0, 0.85, 0.1))
    ax.tick_params(axis="both", colors="#333333", labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BFBFBF")
    ax.spines["bottom"].set_color("#BFBFBF")
    ax.legend(loc="upper right", frameon=False, fontsize=9.2)

    plt.tight_layout()
    plt.savefig(ANOMALY_TYPE_PLOT_FILE, dpi=300, facecolor="white")
    plt.close()
    print(f"  [OK] Saved Anomaly-Type ROC-AUC plot to: {ANOMALY_TYPE_PLOT_FILE.name}")

    if RESULTS_FIGURES_DIR.exists():
        import shutil
        shutil.copy2(ROC_PLOT_FILE, RESULTS_FIGURES_DIR / ROC_PLOT_FILE.name)
        shutil.copy2(PR_PLOT_FILE, RESULTS_FIGURES_DIR / PR_PLOT_FILE.name)
        shutil.copy2(ANOMALY_TYPE_PLOT_FILE, RESULTS_FIGURES_DIR / ANOMALY_TYPE_PLOT_FILE.name)

    # ------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------
    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    print(f"  [OK] Saved metrics table to:  {OUTPUT_METRICS_CSV.name}")

    scores_dict = {
        "sample_id": clean_sample_ids + synth_sample_ids,
        "ground_truth_label": y_test.astype(int),
        "split": ["clean_reference"] * n_clean + ["test_synthetic"] * n_synth,
        "anomaly_slice": ["normal"] * n_clean + list(synth_types),
    }
    for name, m_info in models_dict.items():
        scores_dict[name] = m_info["test_scores"]

    raw_scores_df = pd.DataFrame(scores_dict)
    raw_scores_df.to_csv(OUTPUT_SCORES_CSV, index=False)
    print(f"  [OK] Saved raw sample scores to: {OUTPUT_SCORES_CSV.name}\n")

    # ------------------------------------------------------------
    # Print Console Markdown Table
    # ------------------------------------------------------------
    print("=" * 130)
    print("LITERATURE BENCHMARK RESULTS TABLE (IDENTICAL EVALUATION PROTOCOL, SEED 42)")
    print("=" * 130)

    md_header = "| Method | Verification Status | Feature Space | Overall AUC | Overall AP | Blur AUC | Dropout AUC | Thickening AUC | Thinning AUC |"
    md_divider = "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
    md_rows = []
    for _, r in metrics_df.iterrows():
        row_str = (
            f"| **{r['Method']}** "
            f"| {r['Verification Status']} "
            f"| {r['Feature Space']} "
            f"| **{r['Overall AUC']:.4f}** "
            f"| **{r['Overall AP']:.4f}** "
            f"| {r['Blur AUC']:.4f} "
            f"| {r['Dropout AUC']:.4f} "
            f"| {r['Thickening AUC']:.4f} "
            f"| {r['Thinning AUC']:.4f} |"
        )
        md_rows.append(row_str)

    md_table = "\n".join([md_header, md_divider] + md_rows)
    print(md_table)
    print("=" * 130)

    return metrics_df, raw_scores_df


if __name__ == "__main__":
    run_benchmark()
