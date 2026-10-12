#!/usr/bin/env python3
"""
Positive Control Sanity Audit for Agarwal et al. (2023) 8-D Handcrafted Features + OneClassSVM.

Verifies that the faithful reimplementation of Agarwal et al. (SN Computer Science, 2023)
successfully detects explicit synthetic geometric perturbations (baseline tilt, shear slant,
line spacing corruption, and pen pressure shifts) on pure cropped handwriting regions.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from dataset import load_handwriting_crop
from benchmark_literature_baselines import extract_agarwal_features_single

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv"
AGARWAL_CACHE = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "agarwal_features.npz"


def run_positive_control():
    print("=" * 90)
    print("POSITIVE CONTROL AUDIT: AGARWAL ET AL. (2023) [8-D FEATURES + ONE-CLASS SVM]")
    print("=" * 90)

    if not AGARWAL_CACHE.exists():
        raise FileNotFoundError(f"Missing cache: {AGARWAL_CACHE}. Run benchmark_literature_baselines.py first.")

    data = np.load(AGARWAL_CACHE, allow_pickle=True)
    clean_feats = data["clean_features"]
    synth_feats = data["synth_features"]

    n_clean = len(clean_feats)
    n_synth = len(synth_feats)
    y_eval = np.concatenate([np.zeros(n_clean), np.ones(n_synth)])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    clean_scores = np.zeros(n_clean, dtype=np.float64)
    synth_scores_folds = []

    for tr_i, te_i in kf.split(np.arange(n_clean)):
        scaler = StandardScaler().fit(clean_feats[tr_i])
        ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(scaler.transform(clean_feats[tr_i]))
        clean_scores[te_i] = -ocsvm.decision_function(scaler.transform(clean_feats[te_i]))
        synth_scores_folds.append(-ocsvm.decision_function(scaler.transform(synth_feats)))

    synth_scores = np.mean(synth_scores_folds, axis=0)
    all_scores = np.concatenate([clean_scores, synth_scores])

    roc_auc = roc_auc_score(y_eval, all_scores)
    pr_auc = average_precision_score(y_eval, all_scores)
    print(f"  [Benchmark Verification] Clean N={n_clean}, Synthetic N={n_synth}")
    print(f"  -> Overall ROC-AUC: {roc_auc:.4f}")
    print(f"  -> Overall PR-AUC:  {pr_auc:.4f}")
    print("=" * 90)
    return roc_auc, pr_auc


if __name__ == "__main__":
    run_positive_control()
