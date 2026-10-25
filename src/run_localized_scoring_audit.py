#!/usr/bin/env python3
"""
Scientific Audit & Localized-Scoring Experiments on Corrected Pipeline.

Protocol:
1. Sweep perturbation size (1x, 2x, 4x) and strength (0.25x - 4x);
   Report AUC with bootstrap CI and Strength-0 AUC for:
   - Six-feature OCSVM (nu=0.05, RBF)
   - Six-feature LOF (k=5)
   - Hybrid LOF (k=5, 256-D multimodal embedding)
2. Patch-level scoring:
   - Tile cropped page at native resolution into overlapping 224x224 patches (stride 112 px)
   - Embed with frozen ResNet-50 shared feature path
   - Score against clean patch library (parent and same-writer strictly excluded)
   - Aggregate per page with max and mean of top 5%
   - Compare with page-level holistic score
3. Local features & targeted unit tests:
   - Per-line baseline angle (folded into [-45, 45]), stroke density, and stroke width
   - Targeted unit tests: |z| >= 2.0 = pass on perturbed lines
   - Page aggregation with max and top-k (top-2)
4. Rule 4: A result counts as detection iff Strength-0 AUC in [0.45, 0.55] and 95% bootstrap CI excludes 0.50.
   Report all negative results transparently.
"""

from pathlib import Path
import math
import sys
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from dataset import (
    load_handwriting_crop,
    extract_pixel_features,
    detect_line_bands_from_binary,
    fold_angle_45,
    FORM_ID_MAP,
)
from save_paired_embeddings import CachedFusionModel

# Paths
OUTPUT_DIR = PROJECT_ROOT / "data" / "audit_local"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"
BBOX_CSV = PROJECT_ROOT / "data" / "metadata" / "form_handwriting_bboxes.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "best_dysgraphia_encoder.pth"
AGARWAL_CACHE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "agarwal_features.npz"
EMBEDDINGS_FILE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "paired_embeddings.npz"

BASE_SEED = 42


def bootstrap_auc_ci(y_true, scores, n_bootstraps=1000, alpha=0.05, seed=42):
    """Compute empirical 95% bootstrap confidence interval for ROC-AUC."""
    rng = np.random.RandomState(seed)
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    if not boot_aucs:
        return 0.5, 0.5
    low = np.percentile(boot_aucs, 100 * (alpha / 2))
    high = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return float(low), float(high)


def compute_strength0_auc(scores_clean, n_pseudo_pos=50, seed=42):
    """Compute Strength-0 AUC negative control on clean held-out forms."""
    n = len(scores_clean)
    if n < n_pseudo_pos * 2:
        n_pos = max(1, n // 2)
    else:
        n_pos = n_pseudo_pos
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    s0_y = np.zeros(n)
    s0_y[perm[:n_pos]] = 1.0
    return float(roc_auc_score(s0_y, scores_clean))


# ============================================================
# PERTURBATION GENERATOR (SIZE & STRENGTH SCALING)
# ============================================================

def generate_scaled_perturbation(
    crop_gray: np.ndarray,
    anomaly_type: str,
    size_mult: float,
    strength_mult: float,
    rng: np.random.RandomState,
) -> tuple:
    """
    Generate scaled perturbation strictly within handwriting ink.
    Returns: (perturbed_crop, mask)
    """
    h, w = crop_gray.shape[:2]
    if strength_mult <= 1e-6:
        # Strength 0: negative control, zero perturbation
        return crop_gray.copy(), np.zeros((h, w), dtype=np.uint8)

    # Inverted Otsu to find ink
    inv = 255.0 - crop_gray.astype(np.float32)
    _, otsu = cv2.threshold(inv.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(otsu > 0)
    if len(xs) == 0:
        return crop_gray.copy(), np.zeros((h, w), dtype=np.uint8)

    # Center ink pixel
    c_idx = rng.randint(0, len(xs))
    cx, cy = xs[c_idx], ys[c_idx]

    # Base region dimensions scaled by size_mult
    base_w = rng.randint(int(w * 0.20), int(w * 0.45))
    base_h = rng.randint(int(h * 0.12), int(h * 0.30))
    reg_w = min(w, int(round(base_w * size_mult)))
    reg_h = min(h, int(round(base_h * size_mult)))

    rx1 = max(0, min(cx - reg_w // 2, w - reg_w))
    ry1 = max(0, min(cy - reg_h // 2, h - reg_h))
    rx2 = min(w, rx1 + reg_w)
    ry2 = min(h, ry1 + reg_h)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ry1:ry2, rx1:rx2] = 255
    mask = cv2.bitwise_and(mask, otsu)

    perturbed = crop_gray.copy()
    mask_pixels = np.where(mask > 0)
    if len(mask_pixels[0]) == 0:
        return perturbed, mask

    if anomaly_type == "thickening":
        k_size = max(3, int(round(7 * strength_mult)))
        if k_size % 2 == 0:
            k_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        iters = 2 if strength_mult >= 3.0 else 1
        dil = cv2.dilate(inv.astype(np.uint8), kernel, iterations=iters)
        dil = cv2.GaussianBlur(dil, (3, 3), 0.5)
        thick = 255.0 - dil.astype(np.float32)
        perturbed[mask > 0] = np.clip(thick[mask > 0], 0, 255).astype(np.uint8)

    elif anomaly_type == "thinning":
        k_size = max(3, int(round(3 * max(0.5, strength_mult))))
        if k_size % 2 == 0:
            k_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        eroded = cv2.erode(inv.astype(np.uint8), kernel, iterations=1)
        fade = max(0.1, 1.0 - (0.35 * strength_mult))
        thinned = 255.0 - (eroded.astype(np.float32) * fade)
        perturbed[mask > 0] = np.clip(thinned[mask > 0], 0, 255).astype(np.uint8)

    elif anomaly_type == "blur":
        sigma = 3.0 * strength_mult
        k_size = max(3, int(round(4 * sigma)) | 1)
        blurred = cv2.GaussianBlur(crop_gray, (k_size, k_size), sigma)
        perturbed[mask > 0] = blurred[mask > 0]

    elif anomaly_type == "dropout":
        n_pts = len(mask_pixels[0])
        n_breaks = max(3, int(round(n_pts * 0.006 * strength_mult)))
        n_breaks = min(n_breaks, n_pts)
        chosen = rng.choice(n_pts, n_breaks, replace=False)
        drop_mask = np.ones_like(mask, dtype=bool)
        rx_ell = max(3, int(round(9 * math.sqrt(strength_mult))))
        ry_ell = max(2, int(round(6 * math.sqrt(strength_mult))))
        for idx in chosen:
            px, py = mask_pixels[1][idx], mask_pixels[0][idx]
            cv2.ellipse(drop_mask, (px, py), (rx_ell, ry_ell), rng.randint(0, 180), 0, 360, False, -1)
        paper_bg = int(np.percentile(crop_gray[mask == 0], 70)) if np.any(mask == 0) else 240
        perturbed[~drop_mask & (mask > 0)] = paper_bg
        perturbed = cv2.medianBlur(perturbed, 3)

    return perturbed, mask


# ============================================================
# LINE FEATURE EXTRACTION & STATS
# ============================================================

def extract_line_features_for_crop(crop_gray: np.ndarray) -> list:
    """
    Extract per-line features: baseline_angle, stroke_density, stroke_width.
    Returns list of dicts per line band.
    """
    binary = cv2.adaptiveThreshold(
        crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bands = detect_line_bands_from_binary(cleaned)

    line_records = []
    for li, (y1, y2) in enumerate(bands):
        l_img = cleaned[y1:y2, :]
        ys, xs = np.where(l_img > 0)
        if len(xs) < 20:
            continue
        min_x, max_x = int(xs.min()), int(xs.max())
        line_area = max(1, (max_x - min_x + 1) * (y2 - y1))
        density = float(len(xs) / line_area)

        # Stroke caliper width via distance transform
        dist = cv2.distanceTransform((l_img > 0).astype(np.uint8), cv2.DIST_L2, 5)
        sw = float(2.0 * np.mean(dist[dist > 0])) if np.any(dist > 0) else 0.0

        # Baseline angle via character components OLS
        num_l, _, stats, _ = cv2.connectedComponentsWithStats(l_img, connectivity=8)
        pts = []
        for ci in range(1, num_l):
            cw, ch, ca = (
                stats[ci, cv2.CC_STAT_WIDTH],
                stats[ci, cv2.CC_STAT_HEIGHT],
                stats[ci, cv2.CC_STAT_AREA],
            )
            if ca < 10 or ch < 5 or cw < 2:
                continue
            pts.append((stats[ci, cv2.CC_STAT_LEFT] + cw / 2.0, stats[ci, cv2.CC_STAT_TOP] + ch))

        if len(pts) >= 2:
            xs_p = np.array([p[0] for p in pts], dtype=np.float64)
            ys_p = np.array([p[1] for p in pts], dtype=np.float64)
            denom = float(np.sum((xs_p - np.mean(xs_p)) ** 2))
            if denom > 1e-6:
                slope = float(np.sum((xs_p - np.mean(xs_p)) * (ys_p - np.mean(ys_p))) / denom)
                angle = fold_angle_45(-math.degrees(math.atan(slope)))
            else:
                angle = 0.0
        else:
            angle = 0.0

        line_records.append({
            "line_idx": li,
            "y1": y1,
            "y2": y2,
            "baseline_angle": angle,
            "stroke_density": density,
            "stroke_width": sw,
        })
    return line_records


# ============================================================
# MAIN AUDIT RUNNER
# ============================================================

def run_localized_audit():
    print("=" * 85)
    print("LOCALIZED-SCORING SCIENTIFIC AUDIT ON CORRECTED PIPELINE")
    print("Zero Data Leakage | Validation Split Freezing | Rule 4 Compliance")
    print("=" * 85)

    # 1. Load Metadata & Forms
    print("\n[Step 1/5] Loading metadata, image index, and pretrained checkpoints...")
    meta_df = pd.read_csv(METADATA_FILE)
    form_bboxes_df = pd.read_csv(BBOX_CSV)
    clean_forms = meta_df["form_id"].tolist()
    writer_map = dict(zip(meta_df["form_id"], meta_df["writer_id"]))
    img_path_map = dict(zip(meta_df["form_id"], meta_df["image_path"]))

    print(f"Total clean forms available: {len(clean_forms)}")

    # Load retrained multimodal checkpoint
    ckpt = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    cv_mean = ckpt["feature_mean"].float()
    cv_std = ckpt["feature_std"].float()

    fusion_model = CachedFusionModel()
    fusion_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    fusion_model.eval()

    # Load ResNet-50
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval()

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # 2. Partition Forms: Train vs. Held-Out Evaluation (Disjoint Writers)
    print("\n[Step 2/5] Establishing writer-disjoint partition (Train vs. Held-Out)...")
    unique_writers = sorted(meta_df["writer_id"].unique())
    rng_split = np.random.RandomState(BASE_SEED)
    shuffled_writers = rng_split.permutation(unique_writers)

    n_train_w = int(len(unique_writers) * 0.70)
    train_writers = set(shuffled_writers[:n_train_w])
    heldout_writers = set(shuffled_writers[n_train_w:])

    train_form_ids = [fid for fid in clean_forms if writer_map[fid] in train_writers]
    heldout_form_ids = [fid for fid in clean_forms if writer_map[fid] in heldout_writers]

    print(f"  Train:    {len(train_writers)} writers, {len(train_form_ids)} forms")
    print(f"  Held-Out: {len(heldout_writers)} writers, {len(heldout_form_ids)} forms")

    # Select evaluation cohort: 100 clean forms + 100 perturbed forms
    rng_eval = np.random.RandomState(BASE_SEED + 100)
    eval_clean_ids = list(rng_eval.choice(heldout_form_ids, 100, replace=False))

    # Load crops for evaluation clean forms
    print("  Caching evaluation clean crops...")
    eval_clean_crops = {}
    for fid in eval_clean_ids:
        p = PROJECT_ROOT / img_path_map[fid]
        raw_img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        crop = load_handwriting_crop(fid, raw_img)
        eval_clean_crops[fid] = crop

    # Pre-extract clean reference features from training forms
    print("  Fitting clean training baselines and scalers...")
    ag_cache = np.load(AGARWAL_CACHE, allow_pickle=True)
    fused_cache = np.load(EMBEDDINGS_FILE, allow_pickle=True)

    fid_to_row = {str(r["form_id"]).strip(): i for i, r in meta_df.iterrows()}
    train_row_idx = [fid_to_row[fid] for fid in train_form_ids if fid in fid_to_row]

    train_clean_ag = ag_cache["clean_features"][train_row_idx, :6]
    train_clean_fused = fused_cache["clean_embeddings"][train_row_idx]

    # Fit 6-D Scaler & Models on clean training split
    ag_scaler = StandardScaler().fit(train_clean_ag)
    train_clean_ag_scaled = ag_scaler.transform(train_clean_ag)

    ocsvm_005 = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(train_clean_ag_scaled)
    lof_ag = LocalOutlierFactor(n_neighbors=5, metric="minkowski", novelty=True).fit(train_clean_ag_scaled)
    lof_hybrid = LocalOutlierFactor(n_neighbors=5, metric="minkowski", novelty=True).fit(train_clean_fused)

    # Compute clean scores for evaluation clean forms
    eval_row_idx = [fid_to_row[fid] for fid in eval_clean_ids]
    eval_clean_ag = ag_scaler.transform(ag_cache["clean_features"][eval_row_idx, :6])
    eval_clean_fused = fused_cache["clean_embeddings"][eval_row_idx]

    clean_scores_ocsvm = -ocsvm_005.decision_function(eval_clean_ag)
    clean_scores_lof_ag = -lof_ag.decision_function(eval_clean_ag)
    clean_scores_hybrid = -lof_hybrid.decision_function(eval_clean_fused)

    # Helper function to embed crop with ResNet-50 + Fusion
    def embed_crop_multimodal(crop: np.ndarray, pixel_9d: np.ndarray) -> np.ndarray:
        resized_224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized_224).float() / 255.0
        t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        t = (t - imagenet_mean) / imagenet_std
        with torch.no_grad():
            img_feat = resnet(t)
            cv_t = torch.from_numpy(pixel_9d).float().unsqueeze(0)
            norm_cv = (cv_t - cv_mean) / (cv_std + 1e-8)
            fused = fusion_model(img_feat, norm_cv)
        return fused.cpu().numpy()[0]

    # ============================================================
    # EXPERIMENT 1: PERTURBATION SIZE & STRENGTH SWEEP
    # ============================================================
    print("\n" + "=" * 85)
    print("[EXPERIMENT 1] Multi-Scale Perturbation Size & Strength Sweep")
    print("=" * 85)

    sizes = [1.0, 2.0, 4.0]
    strengths = [0.25, 0.5, 1.0, 2.0, 4.0]
    anomaly_types = ["thickening", "thinning", "blur", "dropout"]

    sweep_records = []
    y_test = np.concatenate([np.zeros(len(eval_clean_ids)), np.ones(len(eval_clean_ids))])

    # First, evaluate Strength 0.0 (Negative Control)
    print("\nEvaluating Strength 0.0 Negative Control...")
    s0_auc_ocsvm = compute_strength0_auc(clean_scores_ocsvm, n_pseudo_pos=50, seed=42)
    s0_auc_lof_ag = compute_strength0_auc(clean_scores_lof_ag, n_pseudo_pos=50, seed=42)
    s0_auc_hybrid = compute_strength0_auc(clean_scores_hybrid, n_pseudo_pos=50, seed=42)

    ci_low_ocsvm, ci_high_ocsvm = bootstrap_auc_ci(
        y_test, np.concatenate([clean_scores_ocsvm, clean_scores_ocsvm]), seed=42
    )
    ci_low_lof, ci_high_lof = bootstrap_auc_ci(
        y_test, np.concatenate([clean_scores_lof_ag, clean_scores_lof_ag]), seed=42
    )
    ci_low_hyb, ci_high_hyb = bootstrap_auc_ci(
        y_test, np.concatenate([clean_scores_hybrid, clean_scores_hybrid]), seed=42
    )

    for m_name, s0_val, ci_l, ci_h in [
        ("Six-Feature OCSVM (nu=0.05)", s0_auc_ocsvm, ci_low_ocsvm, ci_high_ocsvm),
        ("Six-Feature LOF (k=5)", s0_auc_lof_ag, ci_low_lof, ci_high_lof),
        ("Hybrid LOF (k=5)", s0_auc_hybrid, ci_low_hyb, ci_high_hyb),
    ]:
        sweep_records.append({
            "model": m_name,
            "size": "0x",
            "size_mult": 0.0,
            "strength": "0.0x",
            "strength_mult": 0.0,
            "auc": 0.5000,
            "ci_low": ci_l,
            "ci_high": ci_h,
            "strength0_auc": s0_val,
            "rule4_detection": "NEGATIVE CONTROL (N/A)",
        })

    # Sweep grid
    for s_mult in sizes:
        s_lbl = f"{int(s_mult)}x"
        for st_mult in strengths:
            st_lbl = f"{st_mult}x"
            t0 = time.time()
            print(f"  Running Sweep: Size {s_lbl}, Strength {st_lbl}...", end=" ", flush=True)

            synth_ag_feats = []
            synth_fused_feats = []

            for i, fid in enumerate(eval_clean_ids):
                c_crop = eval_clean_crops[fid]
                a_type = anomaly_types[i % len(anomaly_types)]
                rng_p = np.random.RandomState(BASE_SEED * 1000 + int(s_mult * 100) + int(st_mult * 10) + i)

                p_crop, _ = generate_scaled_perturbation(c_crop, a_type, s_mult, st_mult, rng_p)
                p_9d = extract_pixel_features(p_crop)
                p_ag = p_9d[:6]
                p_fused = embed_crop_multimodal(p_crop, p_9d)

                synth_ag_feats.append(p_ag)
                synth_fused_feats.append(p_fused)

            synth_ag_feats = ag_scaler.transform(np.array(synth_ag_feats))
            synth_fused_feats = np.array(synth_fused_feats)

            # Score with models
            sc_s_ocsvm = -ocsvm_005.decision_function(synth_ag_feats)
            sc_s_lof_ag = -lof_ag.decision_function(synth_ag_feats)
            sc_s_hybrid = -lof_hybrid.decision_function(synth_fused_feats)

            # Evaluate each model
            for m_name, sc_c, sc_s, s0_val in [
                ("Six-Feature OCSVM (nu=0.05)", clean_scores_ocsvm, sc_s_ocsvm, s0_auc_ocsvm),
                ("Six-Feature LOF (k=5)", clean_scores_lof_ag, sc_s_lof_ag, s0_auc_lof_ag),
                ("Hybrid LOF (k=5)", clean_scores_hybrid, sc_s_hybrid, s0_auc_hybrid),
            ]:
                full_sc = np.concatenate([sc_c, sc_s])
                auc_val = float(roc_auc_score(y_test, full_sc))
                ci_l, ci_h = bootstrap_auc_ci(y_test, full_sc, seed=42)

                # Rule 4: detection iff strength-0 in [0.45, 0.55] AND CI excludes 0.50 (ci_low > 0.50)
                is_valid_s0 = 0.45 <= s0_val <= 0.55
                excludes_chance = ci_l > 0.50
                is_detection = is_valid_s0 and excludes_chance

                verdict = "PASS" if is_detection else "NEGATIVE RESULT"
                if not is_valid_s0:
                    verdict += " (S0 Fail)"
                elif not excludes_chance:
                    verdict += " (CI Overlaps 0.5)"

                sweep_records.append({
                    "model": m_name,
                    "size": s_lbl,
                    "size_mult": s_mult,
                    "strength": st_lbl,
                    "strength_mult": st_mult,
                    "auc": round(auc_val, 4),
                    "ci_low": round(ci_l, 4),
                    "ci_high": round(ci_h, 4),
                    "strength0_auc": round(s0_val, 4),
                    "rule4_detection": verdict,
                })

            print(f"done ({time.time() - t0:.1f}s)")

    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv(OUTPUT_DIR / "experiment1_perturbation_sweep.csv", index=False)
    print("  Saved experiment1_perturbation_sweep.csv")

    # ============================================================
    # EXPERIMENT 2: PATCH-LEVEL SCORING
    # ============================================================
    print("\n" + "=" * 85)
    print("[EXPERIMENT 2] Patch-Level Scoring (224x224 Patches, ResNet-50)")
    print("Writer & Parent Exclusion | Max & Top 5% Aggregation vs. Page Holistic")
    print("=" * 85)

    # 1. Build clean reference patch library from training forms
    print("  Extracting clean patch reference library from training forms...")
    n_patch_train_forms = 40
    ref_patches = []
    ref_patch_form_ids = []
    ref_patch_writer_ids = []

    for fid in train_form_ids[:n_patch_train_forms]:
        p = PROJECT_ROOT / img_path_map[fid]
        raw_img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        crop = load_handwriting_crop(fid, raw_img)
        ch, cw = crop.shape[:2]
        stride = 112
        p_size = 224

        patch_tensors = []
        for y in range(0, max(1, ch - p_size + 1), stride):
            for x in range(0, max(1, cw - p_size + 1), stride):
                patch = crop[y : y + p_size, x : x + p_size]
                # Filter empty patches (< 50 ink pixels)
                _, otsu_p = cv2.threshold(255 - patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.sum(otsu_p > 0) < 50:
                    continue
                t = torch.from_numpy(patch).float() / 255.0
                t = t.unsqueeze(0).repeat(3, 1, 1)
                patch_tensors.append(t)
                ref_patch_form_ids.append(fid)
                ref_patch_writer_ids.append(writer_map[fid])

        if patch_tensors:
            batch = torch.stack(patch_tensors)
            batch = (batch - imagenet_mean.squeeze(0)) / imagenet_std.squeeze(0)
            with torch.no_grad():
                feats = resnet(batch).cpu().numpy()
            ref_patches.append(feats)

    ref_patches = np.vstack(ref_patches)
    ref_patch_form_ids = np.array(ref_patch_form_ids)
    ref_patch_writer_ids = np.array(ref_patch_writer_ids)
    print(f"  Clean reference patch library: {len(ref_patches)} patches from {n_patch_train_forms} forms")

    # Helper function to extract and embed patches for a crop
    def extract_and_embed_patches(crop: np.ndarray) -> np.ndarray:
        ch, cw = crop.shape[:2]
        stride = 112
        p_size = 224
        patch_tensors = []
        for y in range(0, max(1, ch - p_size + 1), stride):
            for x in range(0, max(1, cw - p_size + 1), stride):
                patch = crop[y : y + p_size, x : x + p_size]
                _, otsu_p = cv2.threshold(255 - patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.sum(otsu_p > 0) < 50:
                    continue
                t = torch.from_numpy(patch).float() / 255.0
                t = t.unsqueeze(0).repeat(3, 1, 1)
                patch_tensors.append(t)
        if not patch_tensors:
            # Fallback to single center resize if no patch has ink
            t = torch.from_numpy(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)).float() / 255.0
            t = t.unsqueeze(0).repeat(3, 1, 1)
            patch_tensors = [t]
        batch = torch.stack(patch_tensors)
        batch = (batch - imagenet_mean.squeeze(0)) / imagenet_std.squeeze(0)
        with torch.no_grad():
            feats = resnet(batch).cpu().numpy()
        return feats

    # Function to score a page's patches against writer-excluded clean reference
    def score_page_patches(test_patches: np.ndarray, test_fid: str, test_wid: int) -> tuple:
        # Strictly exclude parent form and same writer
        valid_mask = (ref_patch_form_ids != test_fid) & (ref_patch_writer_ids != test_wid)
        sub_ref = ref_patches[valid_mask]
        assert len(sub_ref) >= 10, f"Too few reference patches after writer exclusion: {len(sub_ref)}"

        nn_model = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(sub_ref)
        dists, _ = nn_model.kneighbors(test_patches)
        patch_scores = np.mean(dists, axis=1)

        # Max score
        score_max = float(np.max(patch_scores))
        # Top 5% score
        k_top = max(1, int(math.ceil(len(patch_scores) * 0.05)))
        score_top5 = float(np.mean(np.sort(patch_scores)[-k_top:]))
        return score_max, score_top5

    print("  Scoring evaluation cohort (Clean vs. Perturbed)...")
    eval_n = 50  # 50 clean + 50 perturbed
    sub_eval_ids = eval_clean_ids[:eval_n]

    clean_patch_max = []
    clean_patch_top5 = []
    synth_patch_max = []
    synth_patch_top5 = []

    for fid in sub_eval_ids:
        c_crop = eval_clean_crops[fid]
        c_wid = writer_map[fid]

        # Clean page patch scoring
        c_p_feats = extract_and_embed_patches(c_crop)
        c_s_max, c_s_top5 = score_page_patches(c_p_feats, fid, c_wid)
        clean_patch_max.append(c_s_max)
        clean_patch_top5.append(c_s_top5)

        # Perturbed page patch scoring (1x size, 1x strength)
        rng_p = np.random.RandomState(BASE_SEED * 2000 + c_wid)
        p_crop, _ = generate_scaled_perturbation(c_crop, "thickening", 1.0, 1.0, rng_p)
        p_p_feats = extract_and_embed_patches(p_crop)
        p_s_max, p_s_top5 = score_page_patches(p_p_feats, fid, c_wid)
        synth_patch_max.append(p_s_max)
        synth_patch_top5.append(p_s_top5)

    clean_patch_max = np.array(clean_patch_max)
    clean_patch_top5 = np.array(clean_patch_top5)
    synth_patch_max = np.array(synth_patch_max)
    synth_patch_top5 = np.array(synth_patch_top5)

    y_patch = np.concatenate([np.zeros(eval_n), np.ones(eval_n)])

    # Page-level Hybrid LOF on these same 50+50 pages
    page_clean_scores = clean_scores_hybrid[:eval_n]
    # Evaluate synth hybrid on these same 50
    page_synth_hybrid = []
    for fid in sub_eval_ids:
        c_crop = eval_clean_crops[fid]
        rng_p = np.random.RandomState(BASE_SEED * 2000 + writer_map[fid])
        p_crop, _ = generate_scaled_perturbation(c_crop, "thickening", 1.0, 1.0, rng_p)
        p_9d = extract_pixel_features(p_crop)
        p_fused = embed_crop_multimodal(p_crop, p_9d)
        page_synth_hybrid.append(p_fused)
    page_synth_scores = -lof_hybrid.decision_function(np.array(page_synth_hybrid))

    # Evaluate Patch vs Page
    patch_eval_rows = []
    for method_name, sc_c, sc_s in [
        ("Patch-Level Scoring (Max Patch)", clean_patch_max, synth_patch_max),
        ("Patch-Level Scoring (Mean Top 5%)", clean_patch_top5, synth_patch_top5),
        ("Page-Level Holistic Score (Hybrid LOF)", page_clean_scores, page_synth_scores),
    ]:
        full_sc = np.concatenate([sc_c, sc_s])
        auc_v = float(roc_auc_score(y_patch, full_sc))
        ci_l, ci_h = bootstrap_auc_ci(y_patch, full_sc, seed=42)
        s0_v = compute_strength0_auc(sc_c, n_pseudo_pos=25, seed=42)

        is_valid_s0 = 0.45 <= s0_v <= 0.55
        excludes_chance = ci_l > 0.50
        verdict = "PASS" if (is_valid_s0 and excludes_chance) else "NEGATIVE RESULT"
        if not is_valid_s0:
            verdict += " (S0 Fail)"
        elif not excludes_chance:
            verdict += " (CI Overlaps 0.5)"

        patch_eval_rows.append({
            "method": method_name,
            "auc": round(auc_v, 4),
            "ci_low": round(ci_l, 4),
            "ci_high": round(ci_h, 4),
            "strength0_auc": round(s0_v, 4),
            "rule4_detection": verdict,
        })

    patch_df = pd.DataFrame(patch_eval_rows)
    patch_df.to_csv(OUTPUT_DIR / "experiment2_patch_scoring.csv", index=False)
    print("  Saved experiment2_patch_scoring.csv")

    # ============================================================
    # EXPERIMENT 3: LOCAL FEATURES & TARGETED UNIT TESTS
    # ============================================================
    print("\n" + "=" * 85)
    print("[EXPERIMENT 3] Local Line Features & Targeted Unit Tests")
    print("Baseline Angle, Stroke Density, Stroke Width | |z| >= 2.0 Unit Tests")
    print("=" * 85)

    # 1. Fit reference mean and std on clean training lines
    print("  Extracting reference statistics on clean training lines...")
    ref_angles = []
    ref_densities = []
    ref_widths = []

    for fid in train_form_ids[:60]:
        p = PROJECT_ROOT / img_path_map[fid]
        raw_img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        crop = load_handwriting_crop(fid, raw_img)
        lines = extract_line_features_for_crop(crop)
        for ln in lines:
            ref_angles.append(ln["baseline_angle"])
            ref_densities.append(ln["stroke_density"])
            ref_widths.append(ln["stroke_width"])

    mu_angle, sigma_angle = float(np.mean(ref_angles)), float(np.std(ref_angles)) + 1e-6
    mu_dens, sigma_dens = float(np.mean(ref_densities)), float(np.std(ref_densities)) + 1e-6
    mu_width, sigma_width = float(np.mean(ref_widths)), float(np.std(ref_widths)) + 1e-6

    print(f"  Reference Angle:        mu = {mu_angle:.2f} deg, sigma = {sigma_angle:.2f} deg")
    print(f"  Reference Density:      mu = {mu_dens:.4f},      sigma = {sigma_dens:.4f}")
    print(f"  Reference Stroke Width: mu = {mu_width:.2f} px,  sigma = {sigma_width:.2f} px")

    # 2. Targeted Unit Tests on Perturbed Lines (|z| >= 2.0 = pass)
    print("  Running targeted unit tests on perturbed lines (|z| >= 2.0)...")
    unit_test_counts = {t: {"total": 0, "pass_angle": 0, "pass_density": 0, "pass_width": 0, "pass_any": 0} for t in anomaly_types}

    for i, fid in enumerate(sub_eval_ids):
        c_crop = eval_clean_crops[fid]
        a_type = anomaly_types[i % len(anomaly_types)]
        rng_p = np.random.RandomState(BASE_SEED * 3000 + i)
        p_crop, mask = generate_scaled_perturbation(c_crop, a_type, 1.0, 1.0, rng_p)

        # Detect lines on perturbed image
        p_lines = extract_line_features_for_crop(p_crop)
        for ln in p_lines:
            # Check if this line intersects the perturbation mask
            line_mask = mask[ln["y1"] : ln["y2"], :]
            if np.sum(line_mask > 0) < 50:
                continue

            z_ang = abs(ln["baseline_angle"] - mu_angle) / sigma_angle
            z_den = abs(ln["stroke_density"] - mu_dens) / sigma_dens
            z_wid = abs(ln["stroke_width"] - mu_width) / sigma_width

            pass_a = bool(z_ang >= 2.0)
            pass_d = bool(z_den >= 2.0)
            pass_w = bool(z_wid >= 2.0)
            pass_any = pass_a or pass_d or pass_w

            unit_test_counts[a_type]["total"] += 1
            if pass_a:
                unit_test_counts[a_type]["pass_angle"] += 1
            if pass_d:
                unit_test_counts[a_type]["pass_density"] += 1
            if pass_w:
                unit_test_counts[a_type]["pass_width"] += 1
            if pass_any:
                unit_test_counts[a_type]["pass_any"] += 1

    unit_test_rows = []
    for a_type, counts in unit_test_counts.items():
        tot = counts["total"]
        unit_test_rows.append({
            "anomaly_type": a_type,
            "perturbed_lines_tested": tot,
            "pass_angle_pct": round(100.0 * counts["pass_angle"] / tot, 2) if tot > 0 else 0.0,
            "pass_density_pct": round(100.0 * counts["pass_density"] / tot, 2) if tot > 0 else 0.0,
            "pass_width_pct": round(100.0 * counts["pass_width"] / tot, 2) if tot > 0 else 0.0,
            "pass_any_pct (|z|>=2.0)": round(100.0 * counts["pass_any"] / tot, 2) if tot > 0 else 0.0,
        })

    unit_test_df = pd.DataFrame(unit_test_rows)
    unit_test_df.to_csv(OUTPUT_DIR / "experiment3_targeted_unit_tests.csv", index=False)
    print("  Saved experiment3_targeted_unit_tests.csv")

    # 3. Page Aggregation with Max and Top-k
    print("  Evaluating page aggregation (Max line z-score vs. Top-2 lines)...")
    clean_line_max = []
    clean_line_top2 = []
    synth_line_max = []
    synth_line_top2 = []

    def score_crop_lines(crop: np.ndarray) -> tuple:
        lines = extract_line_features_for_crop(crop)
        if not lines:
            return 0.0, 0.0
        line_z = []
        for ln in lines:
            z_ang = abs(ln["baseline_angle"] - mu_angle) / sigma_angle
            z_den = abs(ln["stroke_density"] - mu_dens) / sigma_dens
            z_wid = abs(ln["stroke_width"] - mu_width) / sigma_width
            line_z.append(max(z_ang, z_den, z_wid))
        line_z = np.array(line_z)
        sc_max = float(np.max(line_z))
        k = min(2, len(line_z))
        sc_topk = float(np.mean(np.sort(line_z)[-k:]))
        return sc_max, sc_topk

    for fid in sub_eval_ids:
        c_crop = eval_clean_crops[fid]
        c_max, c_top2 = score_crop_lines(c_crop)
        clean_line_max.append(c_max)
        clean_line_top2.append(c_top2)

        rng_p = np.random.RandomState(BASE_SEED * 4000 + writer_map[fid])
        p_crop, _ = generate_scaled_perturbation(c_crop, "thickening", 1.0, 1.0, rng_p)
        s_max, s_top2 = score_crop_lines(p_crop)
        synth_line_max.append(s_max)
        synth_line_top2.append(s_top2)

    clean_line_max = np.array(clean_line_max)
    clean_line_top2 = np.array(clean_line_top2)
    synth_line_max = np.array(synth_line_max)
    synth_line_top2 = np.array(synth_line_top2)

    y_line = np.concatenate([np.zeros(eval_n), np.ones(eval_n)])

    local_feat_rows = []
    for agg_name, sc_c, sc_s in [
        ("Max Line Anomaly (|z|)", clean_line_max, synth_line_max),
        ("Top-2 Mean Line Anomaly (|z|)", clean_line_top2, synth_line_top2),
    ]:
        full_sc = np.concatenate([sc_c, sc_s])
        auc_v = float(roc_auc_score(y_line, full_sc))
        ci_l, ci_h = bootstrap_auc_ci(y_line, full_sc, seed=42)
        s0_v = compute_strength0_auc(sc_c, n_pseudo_pos=25, seed=42)

        is_valid_s0 = 0.45 <= s0_v <= 0.55
        excludes_chance = ci_l > 0.50
        verdict = "PASS" if (is_valid_s0 and excludes_chance) else "NEGATIVE RESULT"
        if not is_valid_s0:
            verdict += " (S0 Fail)"
        elif not excludes_chance:
            verdict += " (CI Overlaps 0.5)"

        local_feat_rows.append({
            "aggregation": agg_name,
            "auc": round(auc_v, 4),
            "ci_low": round(ci_l, 4),
            "ci_high": round(ci_h, 4),
            "strength0_auc": round(s0_v, 4),
            "rule4_detection": verdict,
        })

    local_feat_df = pd.DataFrame(local_feat_rows)
    local_feat_df.to_csv(OUTPUT_DIR / "experiment3_local_features.csv", index=False)
    print("  Saved experiment3_local_features.csv")

    print("\n" + "=" * 85)
    print("ALL LOCALIZED-SCORING EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    run_localized_audit()
