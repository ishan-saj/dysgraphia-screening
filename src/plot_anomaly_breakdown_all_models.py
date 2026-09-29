"""
Plot Anomaly Type Breakdown across all 5 Competing Models.

Row 1: Example images (Clean + 4 Perturbation Slices: Blur, Dropout, Thickening, Thinning)
Row 2: Grouped bar chart comparing ROC-AUC across all 5 models per perturbation slice.

Outputs:
  data/synthetic_form_anomalies/evaluation_figure.png
"""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "evaluation_figure.png"

# ── Load Metadata & Sample Images ──────────────────────────────
df = pd.read_csv(PROJECT_ROOT / "data" / "synthetic_form_anomalies" / "synthetic_form_anomalies.csv")
meta = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "metadata_all.csv")

clean_path = str(PROJECT_ROOT / meta["image_path"].iloc[0])

samples = {}
for t in ["blur", "dropout", "thickening", "thinning"]:
    row = df[df["anomaly_type"] == t].iloc[0]
    samples[t] = str(PROJECT_ROOT / row["synthetic_image"])

# ── Metrics Data (from literature_benchmark_metrics.csv) ────────
models = [
    "Devi & Kavya\n(2023)",
    "Agarwal\n(2023)",
    "Alevizos\n(2024)",
    "Proposed\nk-NN",
    "Proposed\nLOF (Ours)",
]
colors = ["#E57373", "#FF8A65", "#FFD54F", "#64B5F6", "#81C784"]

data = {
    "blur": [0.5228, 0.4910, 0.5025, 0.5135, 0.6030],
    "dropout": [0.5540, 0.4914, 0.5149, 0.5370, 0.6302],
    "thickening": [0.5411, 0.5282, 0.5211, 0.5569, 0.6610],
    "thinning": [0.4950, 0.5078, 0.5098, 0.5206, 0.6041],
}

corruption_labels = ["Blur", "Dropout", "Thickening", "Thinning"]
corruption_keys = ["blur", "dropout", "thickening", "thinning"]

# ── Figure Layout ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("white")

# Row 1: Images
img_titles = ["Clean (Normal)", "Blur", "Dropout", "Thickening", "Thinning"]
img_paths = [
    clean_path,
    samples["blur"],
    samples["dropout"],
    samples["thickening"],
    samples["thinning"],
]
border_colors = ["#2E7D32", "#1565C0", "#6A1B9A", "#BF360C", "#F9A825"]

for i, (title, path, bc) in enumerate(zip(img_titles, img_paths, border_colors)):
    ax = fig.add_subplot(2, 5, i + 1)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        ax.imshow(img, cmap="gray", aspect="auto")
    else:
        ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_facecolor("#f0f0f0")

    ax.set_title(title, fontsize=11, fontweight="bold", color=bc, pad=6)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(bc)
        spine.set_linewidth(2.5)
        spine.set_visible(True)

# Row 2: Grouped Bar Chart
ax_bar = fig.add_subplot(2, 1, 2)

n_models = len(models)
n_groups = len(corruption_keys)
x = np.arange(n_groups)
width = 0.15
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

for i, (model, color, offset) in enumerate(zip(models, colors, offsets)):
    vals = [data[k][i] for k in corruption_keys]
    bars = ax_bar.bar(
        x + offset,
        vals,
        width=width * 0.92,
        color=color,
        edgecolor="white",
        linewidth=0.8,
        label=model,
    )
    for bar, val in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color="#333333",
        )

ax_bar.axhline(0.5, color="black", linestyle="--", linewidth=1.4, alpha=0.6, label="Random Chance (0.5)")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(corruption_labels, fontsize=12, fontweight="bold")
ax_bar.set_ylabel("ROC-AUC", fontsize=12)
ax_bar.set_title("ROC-AUC by Anomaly Type — All 5 Models", fontsize=13, fontweight="bold", pad=10)
ax_bar.set_ylim(0.40, 0.72)
ax_bar.yaxis.grid(True, alpha=0.35, linestyle="--")
ax_bar.set_axisbelow(True)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=6)

plt.suptitle(
    "Dysgraphia Screening — Synthetic Corruption Samples & 5-Model Benchmark Results",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout(rect=[0, 0, 1, 1])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Generated and saved: {OUTPUT}")
