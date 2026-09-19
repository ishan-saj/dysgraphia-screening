import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic_form_anomalies"
)


# ============================================================
# LOAD RESULTS FROM CSV FILES
# ============================================================

# 5-NN / Proposed anomaly scoring method
proposed_file = DATA_DIR / "anomaly_type_results.csv"

# Isolation Forest
isolation_file = DATA_DIR / "baseline_isolation_forest_results.csv"

# One-Class SVM
ocsvm_file = DATA_DIR / "baseline_ocsvm_results.csv"

# DTE-C
dte_file = DATA_DIR / "baseline_dte_c_results.csv"


print("Loading evaluation results...")


# ------------------------------------------------------------
# 5-NN / Proposed
# ------------------------------------------------------------

proposed_df = pd.read_csv(proposed_file)

proposed_overall = proposed_df[
    proposed_df["anomaly_type"] == "overall"
].iloc[0]

proposed_roc_auc = proposed_overall["roc_auc"]
proposed_ap = proposed_overall["average_precision"]


# ------------------------------------------------------------
# Isolation Forest
# ------------------------------------------------------------

isolation_df = pd.read_csv(isolation_file)

isolation_overall = isolation_df[
    isolation_df["anomaly_type"] == "overall"
].iloc[0]

isolation_roc_auc = isolation_overall["roc_auc"]
isolation_ap = isolation_overall["average_precision"]


# ------------------------------------------------------------
# OCSVM
# ------------------------------------------------------------

ocsvm_df = pd.read_csv(ocsvm_file)

ocsvm_overall = ocsvm_df[
    ocsvm_df["anomaly_type"] == "overall"
].iloc[0]

ocsvm_roc_auc = ocsvm_overall["roc_auc"]
ocsvm_ap = ocsvm_overall["average_precision"]


# ------------------------------------------------------------
# DTE-C
# ------------------------------------------------------------

dte_df = pd.read_csv(dte_file)

dte_overall = dte_df[
    dte_df["anomaly_type"] == "overall"
].iloc[0]

dte_roc_auc = dte_overall["roc_auc"]
dte_ap = dte_overall["average_precision"]


# ============================================================
# COLLECT RESULTS
# ============================================================

methods = [
    "OCSVM",
    "Isolation Forest",
    "5-NN",
    "DTE-C"
]

roc_auc = [
    ocsvm_roc_auc,
    isolation_roc_auc,
    proposed_roc_auc,
    dte_roc_auc
]

average_precision = [
    ocsvm_ap,
    isolation_ap,
    proposed_ap,
    dte_ap
]


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nCurrent results:")
print("-" * 50)

for method, auc, ap in zip(
    methods,
    roc_auc,
    average_precision
):
    print(
        f"{method:18s} "
        f"ROC-AUC = {auc:.4f}   "
        f"AP = {ap:.4f}"
    )


# ============================================================
# CREATE PLOT
# ============================================================

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

bars1 = ax.bar(
    x - width / 2,
    roc_auc,
    width,
    label="ROC-AUC"
)

bars2 = ax.bar(
    x + width / 2,
    average_precision,
    width,
    label="Average Precision"
)


# ============================================================
# LABELS
# ============================================================

ax.set_xlabel("Method")
ax.set_ylabel("Score")

ax.set_title(
    "Comparison of Anomaly Detection Methods"
)

ax.set_xticks(x)
ax.set_xticklabels(methods)

ax.set_ylim(0, 0.75)

ax.legend()


# ============================================================
# VALUE LABELS
# ============================================================

for bars in [bars1, bars2]:

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.4f}",
            ha="center",
            va="bottom"
        )


# ============================================================
# SAVE
# ============================================================

output_file = (
    DATA_DIR
    / "method_comparison.png"
)

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nPlot saved to:")
print(output_file)

plt.show()