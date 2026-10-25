#!/usr/bin/env python3
"""
Audit & Re-run of Devi & Kavya (2023) Baseline on Corrected Pipeline.

Checks:
1. Kekre matrix normalization:
   - Evaluates both Unnormalized and Orthonormalized Kekre (8x8).
2. StandardScaler fit strictly on training folds.
3. Exclusions:
   - Parent form exclusion.
   - Strict protocol: Parent + Same-Writer exclusion.
4. Scorer:
   - Mean of 5-NN distances vs 5th-NN distance.
5. Metrics:
   - ROC-AUC
   - 1,000-resample Bootstrap 95% CI
   - Strength-0 Negative Control AUC
"""

from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
from scipy.fftpack import dct
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"
PAIRED_CSV = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "paired_clean_vs_synthetic.csv"
SYNTHETIC_CSV = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "synthetic_form_anomalies.csv"
EXISTING_CACHE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "devi_kavya_features.npz"

from dataset import load_handwriting_crop, resize_handwriting_crop, FORM_ID_MAP


def generate_unnormalized_kekre_matrix(N=8):
    """Kekre matrix with subdiagonal = -(N - i)."""
    K = np.zeros((N, N), dtype=np.float32)
    K[0, :] = 1.0
    for i in range(1, N):
        K[i, i - 1] = -(N - i)
        K[i, i:] = 1.0
    return K


def generate_orthonormal_kekre_matrix(N=8):
    """Kekre matrix normalized row-by-row so that K @ K.T = I."""
    K = generate_unnormalized_kekre_matrix(N)
    norms = np.linalg.norm(K, axis=1, keepdims=True)
    return K / norms


def extract_kekre_dct_128(crop_gray: np.ndarray, K_mat: np.ndarray) -> np.ndarray:
    resized_u8 = resize_handwriting_crop(crop_gray, size=224)
    img_224 = resized_u8.astype(np.float32) / 255.0
    h, w = img_224.shape
    blocks = img_224.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
    kekre_blocks = K_mat @ blocks @ K_mat.T
    kdct = dct(dct(kekre_blocks, axis=2, norm="ortho"), axis=1, norm="ortho")
    kdct_flat = kdct.reshape(blocks.shape[0], 64)
    return np.concatenate([kdct_flat.mean(axis=0), kdct_flat.std(axis=0)]).astype(np.float32)


def bootstrap_ci(y_true, scores, n_bootstraps=1000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], scores[idx]))
    low = np.percentile(boot_aucs, 100 * (alpha / 2))
    high = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return float(low), float(high)


def compute_strength0_auc(scores_clean, n_pseudo_pos=500, seed=42):
    n = len(scores_clean)
    n_pos = min(n_pseudo_pos, n // 2)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    s0_y = np.zeros(n)
    s0_y[perm[:n_pos]] = 1.0
    return float(roc_auc_score(s0_y, scores_clean))


def run_devi_kavya_audit():
    print("=" * 80)
    print("DEVI & KAVYA (2023) RETEST & KEKRE NORMALIZATION AUDIT")
    print("=" * 80)

    # 1. Load Metadata
    meta_df = pd.read_csv(METADATA_FILE)
    paired_df = pd.read_csv(PAIRED_CSV)
    synth_meta_df = pd.read_csv(SYNTHETIC_CSV)

    clean_form_ids = [str(x).strip() for x in meta_df["form_id"].tolist()]
    clean_writers = meta_df["writer_id"].tolist()
    clean_paths = meta_df["image_path"].tolist()

    fid_to_idx = {fid: i for i, fid in enumerate(clean_form_ids)}
    fid_to_writer = {fid: w for fid, w in zip(clean_form_ids, clean_writers)}

    synth_source_forms = [
        FORM_ID_MAP.get(str(x).strip(), str(x).strip())
        for x in synth_meta_df["source_form"].tolist()
    ]
    synth_paths = synth_meta_df["synthetic_image"].tolist()
    synth_types = paired_df["anomaly_type"].tolist()

    synth_parent_idx = np.array([fid_to_idx[sf] for sf in synth_source_forms], dtype=int)
    synth_writers = np.array([fid_to_writer[sf] for sf in synth_source_forms])

    n_clean = len(clean_form_ids)
    n_synth = len(synth_source_forms)
    y_test = np.concatenate([np.zeros(n_clean), np.ones(n_synth)])

    # 2. Load Existing Cache (for ResNet-18 features 512-D)
    print("\nLoading cached ResNet-18 512-D features...")
    old_cache = np.load(EXISTING_CACHE, allow_pickle=True)
    clean_r18 = old_cache["clean_features"][:, 128:]
    synth_r18 = old_cache["synth_features"][:, 128:]
    old_clean_k128 = old_cache["clean_features"][:, :128]
    old_synth_k128 = old_cache["synth_features"][:, :128]

    # 3. Compute Orthonormal Kekre-DCT 128-D features
    print("Extracting Orthonormal Kekre-DCT (8x8, normalized rows) 128-D features...")
    K_ortho = generate_orthonormal_kekre_matrix(8)
    K_unnorm = generate_unnormalized_kekre_matrix(8)

    t0 = time.time()
    ortho_clean_k128 = []
    for p, fid in zip(clean_paths, clean_form_ids):
        raw = cv2.imread(str(PROJECT_ROOT / p), cv2.IMREAD_GRAYSCALE)
        crop = load_handwriting_crop(fid, raw)
        ortho_clean_k128.append(extract_kekre_dct_128(crop, K_ortho))
    ortho_clean_k128 = np.array(ortho_clean_k128, dtype=np.float32)

    ortho_synth_k128 = []
    for p, sfid in zip(synth_paths, synth_source_forms):
        raw = cv2.imread(str(PROJECT_ROOT / p), cv2.IMREAD_GRAYSCALE)
        crop = load_handwriting_crop(sfid, raw)
        ortho_synth_k128.append(extract_kekre_dct_128(crop, K_ortho))
    ortho_synth_k128 = np.array(ortho_synth_k128, dtype=np.float32)
    print(f"Extraction completed in {time.time() - t0:.1f} s")

    # Combine 128 + 512 = 640-D
    feats_ortho_clean = np.hstack([ortho_clean_k128, clean_r18])
    feats_ortho_synth = np.hstack([ortho_synth_k128, synth_r18])

    feats_unnorm_clean = np.hstack([old_clean_k128, clean_r18])
    feats_unnorm_synth = np.hstack([old_synth_k128, synth_r18])

    # 4. Cross-Validation Setup (5-Fold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(np.arange(n_clean)))

    # Function to evaluate k-NN with specific exclusion protocol
    def evaluate_knn(
        clean_X, synth_X, standardize=True, score_type="mean", strict_writer_exclusion=False
    ):
        clean_scores = np.zeros(n_clean, dtype=np.float64)
        synth_scores = np.zeros(n_synth, dtype=np.float64)

        for tr_i, te_i in folds:
            te_set = set(int(x) for x in te_i)
            s_mask = np.array([int(p) in te_set for p in synth_parent_idx], dtype=bool)

            c_tr = clean_X[tr_i]
            c_te = clean_X[te_i]
            s_te = synth_X[s_mask]

            if standardize:
                scaler = StandardScaler().fit(c_tr)
                c_tr = scaler.transform(c_tr)
                c_te = scaler.transform(c_te)
                s_te = scaler.transform(s_te)

            if not strict_writer_exclusion:
                # Parent exclusion: for test synthetic, parent index is in te_i (not in tr_i)
                knn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(c_tr)
                dc, _ = knn.kneighbors(c_te)
                ds, _ = knn.kneighbors(s_te)

                if score_type == "mean":
                    clean_scores[te_i] = dc.mean(axis=1)
                    synth_scores[s_mask] = ds.mean(axis=1)
                else:  # 5th NN
                    clean_scores[te_i] = dc[:, 4]
                    synth_scores[s_mask] = ds[:, 4]
            else:
                # Strict: exclude same writer for each test form
                tr_writers = np.array([clean_writers[i] for i in tr_i])
                s_te_writers = synth_writers[s_mask]

                # For clean test
                for local_idx, orig_idx in enumerate(te_i):
                    w = clean_writers[orig_idx]
                    allowed_tr = np.where(tr_writers != w)[0]
                    knn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(c_tr[allowed_tr])
                    d, _ = knn.kneighbors(c_te[local_idx : local_idx + 1])
                    clean_scores[orig_idx] = d.mean() if score_type == "mean" else d[0, 4]

                # For synth test
                synth_sub_indices = np.where(s_mask)[0]
                for local_idx, orig_idx in enumerate(synth_sub_indices):
                    w = s_te_writers[local_idx]
                    allowed_tr = np.where(tr_writers != w)[0]
                    knn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(c_tr[allowed_tr])
                    d, _ = knn.kneighbors(s_te[local_idx : local_idx + 1])
                    synth_scores[orig_idx] = d.mean() if score_type == "mean" else d[0, 4]

        scores = np.concatenate([clean_scores, synth_scores])
        auc_v = float(roc_auc_score(y_test, scores))
        ap_v = float(average_precision_score(y_test, scores))
        ci_l, ci_h = bootstrap_ci(y_test, scores, n_bootstraps=1000, seed=42)
        s0_auc = compute_strength0_auc(clean_scores, n_pseudo_pos=500, seed=42)
        return {
            "auc": auc_v,
            "ap": ap_v,
            "ci_low": ci_l,
            "ci_high": ci_h,
            "strength0_auc": s0_auc,
            "clean_scores": clean_scores,
            "synth_scores": synth_scores,
        }

    # Run variations
    experiments = [
        # 1. Orthonormal Kekre + StandardScaler + Mean 5-NN + Parent Exclusion
        ("Orthonormal Kekre (8x8) | StandardScaler | Mean 5-NN | Parent Excl", feats_ortho_clean, feats_ortho_synth, True, "mean", False),
        # 2. Orthonormal Kekre (8x8) | StandardScaler | 5th-NN | Parent Excl
        ("Orthonormal Kekre (8x8) | StandardScaler | 5th-NN   | Parent Excl", feats_ortho_clean, feats_ortho_synth, True, "5th", False),
        # 3. Orthonormal Kekre (8x8) | StandardScaler | Mean 5-NN | Strict Writer Excl
        ("Orthonormal Kekre (8x8) | StandardScaler | Mean 5-NN | Strict Writer Excl", feats_ortho_clean, feats_ortho_synth, True, "mean", True),
        # 4. Unnormalized Kekre (8x8) | StandardScaler | Mean 5-NN | Parent Excl
        ("Unnormalized Kekre (8x8) | StandardScaler | Mean 5-NN | Parent Excl", feats_unnorm_clean, feats_unnorm_synth, True, "mean", False),
        # 5. Orthonormal Kekre (8x8) | Plain Euclidean (NO Scaler) | Mean 5-NN
        ("Orthonormal Kekre (8x8) | Raw Euclidean  | Mean 5-NN | Parent Excl", feats_ortho_clean, feats_ortho_synth, False, "mean", False),
    ]

    results = []
    print("\nRunning experimental variations...")
    for label, c_feats, s_feats, std_flag, sc_type, strict_flag in experiments:
        t_start = time.time()
        res = evaluate_knn(c_feats, s_feats, standardize=std_flag, score_type=sc_type, strict_writer_exclusion=strict_flag)
        dt = time.time() - t_start
        print(f"  {label}: AUC = {res['auc']:.4f} (95% CI: [{res['ci_low']:.4f}, {res['ci_high']:.4f}]), S0-AUC = {res['strength0_auc']:.4f}, AP = {res['ap']:.4f} ({dt:.1f}s)")
        results.append({
            "Configuration": label,
            "Kekre Matrix": "Orthonormal (8x8)" if "Orthonormal" in label else "Unnormalized (8x8)",
            "Scaling": "StandardScaler (Train folds only)" if std_flag else "Plain Euclidean (None)",
            "k-NN Metric": "Mean of 5-NN" if sc_type == "mean" else "5th-NN distance",
            "Exclusions": "Parent + Same-Writer" if strict_flag else "Parent Form",
            "ROC-AUC": round(res["auc"], 4),
            "95% Bootstrap CI": f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]",
            "Strength-0 AUC": round(res["strength0_auc"], 4),
            "Average Precision": round(res["ap"], 4),
        })

    # Slice evaluation on the primary configuration: Orthonormal + StandardScaler + Mean 5-NN + Parent Excl
    primary_res = evaluate_knn(feats_ortho_clean, feats_ortho_synth, standardize=True, score_type="mean", strict_writer_exclusion=False)
    slices = ["blur", "dropout", "thickening", "thinning"]
    slice_aucs = {}
    for s_name in slices:
        s_mask = np.array([st == s_name for st in synth_types], dtype=bool)
        sc_slice = np.concatenate([primary_res["clean_scores"], primary_res["synth_scores"][s_mask]])
        y_slice = np.concatenate([np.zeros(n_clean), np.ones(np.sum(s_mask))])
        slice_aucs[s_name] = round(float(roc_auc_score(y_slice, sc_slice)), 4)
    print("\nPer-Slice ROC-AUC (Primary Orthonormal Config):", slice_aucs)

    df_res = pd.DataFrame(results)
    out_csv = PROJECT_ROOT / "data" / "audit_local" / "devi_kavya_rerun_metrics.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved rerun metrics to: {out_csv}")


if __name__ == "__main__":
    run_devi_kavya_audit()
