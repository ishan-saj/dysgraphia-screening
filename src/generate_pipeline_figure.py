#!/usr/bin/env python3
"""
Generate publication-quality diagram of the 5 parallel benchmark pipelines:
Matches Slide 15 layout with all verified corrections, parameters, and benchmark metrics.
Saves high-DPI PNG and SVG to results/figures/.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "benchmark_pipeline_architecture.png"
SVG_PATH = OUTPUT_DIR / "benchmark_pipeline_architecture.svg"

# Setup high-resolution figure (16:9 widescreen presentation format)
fig, ax = plt.subplots(figsize=(18, 10.2), dpi=300)
ax.set_xlim(0, 1800)
ax.set_ylim(0, 1020)
ax.axis("off")

# Theme Colors matching the slide
BG_COLOR = "#FFFFFF"
TOP_BOX_BG = "#FFF7E6"
TOP_BOX_BORDER = "#F5B041"
BOTTOM_BOX_BG = "#FFF7E6"
BOTTOM_BOX_BORDER = "#F5B041"

BLUE_CARD_BG = "#EFF6FF"
BLUE_CARD_BORDER = "#60A5FA"
BLUE_HEADER = "#1D4ED8"

GREEN_CARD_BG = "#F0FDF4"
GREEN_CARD_BORDER = "#4ADE80"
GREEN_HEADER = "#15803D"

ARROW_COLOR = "#94A3B8"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#475569"

def draw_rounded_rect(ax, x, y, w, h, bg, border, radius=12, lw=1.6, zorder=2):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg,
        edgecolor=border,
        linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(rect)

def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=1.8):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=14,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=3
    )

# -------------------------------------------------------------
# 1. MAIN TITLE
# -------------------------------------------------------------
ax.text(
    900, 985,
    "Experimental Setup: Shared Reference Population, Five Parallel Anomaly-Scoring Pipelines",
    ha="center", va="center",
    fontsize=16, fontweight="bold", color=TEXT_DARK
)

# -------------------------------------------------------------
# 2. TOP GLOBAL BOX: IAM Dataset & Perturbation Suite
# -------------------------------------------------------------
draw_rounded_rect(ax, 280, 850, 1240, 95, TOP_BOX_BG, TOP_BOX_BORDER, radius=14, lw=1.8)
ax.text(
    900, 915,
    "IAM Handwriting Reference & Synthetic Clinical Perturbation Suite",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color="#B45309"
)
ax.text(
    900, 890,
    "1,539 Clean Normal Reference Forms  |  1,000 Synthetic Handwriting Anomalies",
    ha="center", va="center",
    fontsize=11, fontweight="semibold", color="#78350F"
)
ax.text(
    900, 868,
    "Strictly Bounded to Handwriting Ink Region: Blur (Tremor)  •  Dropout (Pen-Lift)  •  Thickening (Hypertonia)  •  Thinning (Hypotonia)",
    ha="center", va="center",
    fontsize=9.5, style="italic", color="#92400E"
)

# Arrow from Top Box down to columns
draw_arrow(ax, 900, 850, 900, 815)

# -------------------------------------------------------------
# 3. FIVE PARALLEL PIPELINE COLUMNS
# -------------------------------------------------------------
col_width = 300
gap = 25
start_x = (1800 - (5 * col_width + 4 * gap)) / 2  # 100

columns_data = [
    {
        "title": "Devi & Kavya (2023)",
        "subtitle": "Adapted Frequency Baseline",
        "type": "literature",
        "blocks": [
            ("Input Form", "Cropped Handwriting Image"),
            ("Feature Extraction", "8×8 Kekre-DCT (128-D)\n+ ResNet-18 (512-D)"),
            ("Representation", "640-D Fused Vector\n(Frequency-Visual Space)"),
            ("Anomaly Scorer", "Euclidean k-NN Distance\n(k = 5, Parent-Excluded)"),
            ("Anomaly Score", "ROC-AUC: 0.5842\nAverage Precision: 0.4677"),
        ]
    },
    {
        "title": "Agarwal et al. (2023)",
        "subtitle": "Faithful Reimplementation",
        "type": "literature",
        "blocks": [
            ("Input Form", "Cropped Handwriting Image"),
            ("Feature Extraction", "8 Handcrafted Features\n(baseline, slant, pressure, spacing, etc.)"),
            ("Representation", "Standardized Feature Vector\n(8-D Geometric Space)"),
            ("Anomaly Scorer", "One-Class SVM\n(RBF Kernel, ν = 0.05, γ = scale)"),
            ("Anomaly Score", "ROC-AUC: 0.6218\nAverage Precision: 0.5123"),
        ]
    },
    {
        "title": "Alevizos et al. (2024)",
        "subtitle": "Adapted Sequence Autoencoder",
        "type": "literature",
        "blocks": [
            ("Input Form", "Cropped Handwriting Image"),
            ("Feature Extraction", "6 Pixel-Extracted Dynamics\n(deviation, spacing, density, slant, width, height)"),
            ("Representation", "Conv1D Encoder → LSTM\n(Latent Bottleneck = 32-D / 16-D)"),
            ("Anomaly Scorer", "Conv1D-LSTM Decoder\n(L2 Reconstruction Error)"),
            ("Anomaly Score", "ROC-AUC: 0.6584\nAverage Precision: 0.5633"),
        ]
    },
    {
        "title": "Proposed — k-NN Ablation",
        "subtitle": "Multimodal Architecture Ablation",
        "type": "proposed",
        "blocks": [
            ("Input Form", "Dual Input:\nFull Form + Native Crop"),
            ("Feature Extraction", "ResNet-50 Visual (2048-D)\n+ OpenCV Dynamics (9-D → 64-D)"),
            ("Representation", "256-D Fused Embedding\n(Channel Attention Projection)"),
            ("Anomaly Scorer", "Euclidean k-NN Distance\n(k = 5, Parent-Excluded)"),
            ("Anomaly Score", "ROC-AUC: 0.7125\nAverage Precision: 0.5739"),
        ]
    },
    {
        "title": "Proposed — Adaptive LOF (Ours)",
        "subtitle": "Full Production Architecture",
        "type": "proposed",
        "blocks": [
            ("Input Form", "Dual Input:\nFull Form + Native Crop"),
            ("Feature Extraction", "ResNet-50 Visual (2048-D)\n+ OpenCV Dynamics (9-D → 64-D)"),
            ("Representation", "256-D Fused Embedding\n(Channel Attention Projection)"),
            ("Anomaly Scorer", "Adaptive Local Outlier Factor\n(LOF, k = 5, Novelty Mode)"),
            ("Anomaly Score", "ROC-AUC: 0.7714\nAverage Precision: 0.7348"),
        ]
    },
]

row_y = [730, 615, 500, 385, 270]
row_h = [75, 80, 80, 80, 80]

for c_idx, col in enumerate(columns_data):
    cx = start_x + c_idx * (col_width + gap)
    is_lit = (col["type"] == "literature")
    header_color = BLUE_HEADER if is_lit else GREEN_HEADER
    card_bg = BLUE_CARD_BG if is_lit else GREEN_CARD_BG
    card_border = BLUE_CARD_BORDER if is_lit else GREEN_CARD_BORDER

    # Column Header Title
    ax.text(
        cx + col_width / 2, 815,
        col["title"],
        ha="center", va="bottom",
        fontsize=11.5, fontweight="bold", color=header_color
    )
    ax.text(
        cx + col_width / 2, 802,
        col["subtitle"],
        ha="center", va="top",
        fontsize=8.5, color=TEXT_MUTED
    )

    # Draw 5 blocks per column
    for r_idx, (b_title, b_desc) in enumerate(col["blocks"]):
        by = row_y[r_idx]
        bh = row_h[r_idx]

        is_score_block = (r_idx == 4)
        if is_score_block:
            # Highlight score block with slightly darker accent tint
            block_bg = "#DBEAFE" if is_lit else "#DCFCE7"
            block_border = "#3B82F6" if is_lit else "#22C55E"
            lw = 2.0
        else:
            block_bg = card_bg
            block_border = card_border
            lw = 1.4

        draw_rounded_rect(ax, cx, by, col_width, bh, block_bg, block_border, radius=10, lw=lw)

        # Text in block
        if is_score_block:
            ax.text(
                cx + col_width / 2, by + bh - 16,
                b_title,
                ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=header_color
            )
            ax.text(
                cx + col_width / 2, by + 16,
                b_desc,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT_DARK
            )
        else:
            ax.text(
                cx + col_width / 2, by + bh - 14,
                b_title,
                ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=header_color
            )
            ax.text(
                cx + col_width / 2, by + 12,
                b_desc,
                ha="center", va="bottom",
                fontsize=8.5, color=TEXT_DARK
            )

        # Arrow to next block in column
        if r_idx < 4:
            draw_arrow(ax, cx + col_width / 2, by, cx + col_width / 2, row_y[r_idx + 1] + row_h[r_idx + 1])

    # Arrow from last block to bottom evaluation block
    draw_arrow(ax, cx + col_width / 2, row_y[4], cx + col_width / 2, 195)

# -------------------------------------------------------------
# 4. BOTTOM GLOBAL BOX: Evaluation Protocol
# -------------------------------------------------------------
draw_rounded_rect(ax, 280, 100, 1240, 95, BOTTOM_BOX_BG, BOTTOM_BOX_BORDER, radius=14, lw=1.8)
ax.text(
    900, 168,
    "Unified Evaluation Protocol: Parent-Form Excluded 5-Fold Cross-Fitting",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color="#B45309"
)
ax.text(
    900, 143,
    "Zero-Leakage Partition: Synthetic anomaly forms never tested against models fit on their clean source forms",
    ha="center", va="center",
    fontsize=10.5, color="#78350F"
)
ax.text(
    900, 120,
    "Reported Evaluation Metrics: Overall ROC-AUC  •  Overall Average Precision (AP / PR-AUC)  •  Per-Corruption Slices (Blur • Dropout • Thickening • Thinning)",
    ha="center", va="center",
    fontsize=9.5, fontweight="semibold", color="#92400E"
)

# -------------------------------------------------------------
# 5. LEGEND & FOOTER
# -------------------------------------------------------------
legend_y = 45
draw_rounded_rect(ax, 590, legend_y - 12, 28, 20, BLUE_CARD_BG, BLUE_CARD_BORDER, radius=5, lw=1.4)
ax.text(630, legend_y, "Literature Baselines (Adapted / Faithful Reimplementations)", ha="left", va="center", fontsize=9.5, color=TEXT_MUTED)

draw_rounded_rect(ax, 1060, legend_y - 12, 28, 20, GREEN_CARD_BG, GREEN_CARD_BORDER, radius=5, lw=1.4)
ax.text(1100, legend_y, "Proposed Multimodal Architectures (Zero-Shot Density Scorers)", ha="left", va="center", fontsize=9.5, color=TEXT_MUTED)

# Slide Number Badge
ax.text(1750, 45, "15", ha="right", va="center", fontsize=16, fontweight="bold", color="#64748B")

plt.tight_layout()
plt.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
plt.savefig(SVG_PATH, format="svg", bbox_inches="tight", facecolor=BG_COLOR)
plt.close()

print(f"Publication figure generated successfully:")
print(f"  PNG: {PNG_PATH}")
print(f"  SVG: {SVG_PATH}")
