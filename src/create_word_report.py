from pathlib import Path
import pandas as pd
import html


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANOMALY_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "p06-052_word_anomaly.csv"
)

EXPLANATION_CSV = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "p06-052_word_explanation.csv"
)

OUTPUT_HTML = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "p06-052_word_anomaly_report.html"
)

TOP_K = 10


# ============================================================
# LOAD DATA
# ============================================================

if not ANOMALY_CSV.exists():
    raise FileNotFoundError(f"Missing:\n{ANOMALY_CSV}")

if not EXPLANATION_CSV.exists():
    raise FileNotFoundError(f"Missing:\n{EXPLANATION_CSV}")

anomaly_df = pd.read_csv(ANOMALY_CSV)
explanation_df = pd.read_csv(EXPLANATION_CSV)

anomaly_df = anomaly_df.sort_values(
    "word_anomaly_score",
    ascending=False
).head(TOP_K)

explanation_df = explanation_df.sort_values(
    "rank"
).head(TOP_K)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "aspect_ratio",
    "ink_aspect_ratio",
    "ink_density",
    "centroid_x_ratio",
    "centroid_y_ratio",
    "largest_component_ratio",
    "horizontal_projection_std",
    "vertical_projection_std",
    "lower_ink_ratio",
]


# ============================================================
# BUILD FEATURE TABLE
# ============================================================

feature_rows = ""

for _, row in anomaly_df.iterrows():

    rank = int(
        explanation_df.loc[
            explanation_df["word_id"] == row["word_id"],
            "rank"
        ].iloc[0]
    ) if row["word_id"] in explanation_df["word_id"].values else ""

    feature_cells = ""

    for feature in FEATURES:

        z_column = f"{feature}_z"

        if z_column in row.index:

            value = row[z_column]

            if pd.notna(value):

                feature_cells += (
                    f"<td>{float(value):.2f}</td>"
                )

            else:
                feature_cells += "<td>-</td>"

        else:
            feature_cells += "<td>-</td>"

    feature_rows += f"""
    <tr>
        <td><b>#{rank}</b></td>
        <td><b>{html.escape(str(row['transcription']))}</b></td>
        <td>{float(row['word_anomaly_score']):.3f}</td>
        {feature_cells}
        <td>
            {html.escape(str(row['main_anomaly_feature']))}
        </td>
    </tr>
    """


# ============================================================
# EXPLANATION CARDS
# ============================================================

cards = ""

for _, row in explanation_df.iterrows():

    word = html.escape(str(row["transcription"]))
    rank = int(row["rank"])
    score = float(row["word_anomaly_score"])

    explanation = html.escape(
        str(row["explanation"])
    )

    main_feature = html.escape(
        str(row["main_feature"])
    )

    main_z = float(row["main_feature_z"])

    cards += f"""
    <div class="card">

        <div class="card-header">
            <span class="rank">#{rank}</span>
            <span class="word">{word}</span>
            <span class="score">
                Score: {score:.3f}
            </span>
        </div>

        <p>
            <b>Main contributing feature:</b>
            {main_feature}
            (z = {main_z:.2f})
        </p>

        <p>
            {explanation}
        </p>

    </div>
    """


# ============================================================
# HTML DOCUMENT
# ============================================================

html_document = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>
p06-052 Word-Level Anomaly Report
</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    background: #f5f5f5;
    color: #222;
}}

h1 {{
    margin-bottom: 5px;
}}

.subtitle {{
    color: #666;
    margin-bottom: 30px;
}}

.summary {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 25px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background: white;
    font-size: 13px;
}}

th {{
    background: #eeeeee;
    padding: 10px;
    border: 1px solid #cccccc;
}}

td {{
    padding: 8px;
    border: 1px solid #dddddd;
    text-align: center;
}}

.card {{
    background: white;
    margin-top: 15px;
    padding: 18px;
    border-radius: 8px;
    border-left: 5px solid #555;
}}

.card-header {{
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 10px;
}}

.rank {{
    font-size: 18px;
    font-weight: bold;
}}

.word {{
    font-size: 20px;
    font-weight: bold;
}}

.score {{
    margin-left: auto;
    font-weight: bold;
}}

.note {{
    background: #fff8dc;
    padding: 15px;
    border-radius: 8px;
    margin-top: 25px;
}}

</style>

</head>

<body>

<h1>
p06-052 Word-Level Anomaly Analysis
</h1>

<div class="subtitle">
Dataset-relative handwriting unusualness analysis
</div>

<div class="summary">

<h2>Summary</h2>

<p>
<b>Target sample:</b> p06-052
</p>

<p>
<b>Total handwriting words analyzed:</b> 32
</p>

<p>
<b>Reference words:</b> 10,000 IAM handwriting words
</p>

<p>
<b>Features used:</b> 9
</p>

<p>
The anomaly score represents relative statistical unusualness
compared with the reference word distribution. It is not a
clinical dysgraphia probability or diagnosis.
</p>

</div>


<h2>Top 10 Anomalous Words</h2>

<table>

<tr>
    <th>Rank</th>
    <th>Word</th>
    <th>Score</th>
    <th>Aspect</th>
    <th>Ink Aspect</th>
    <th>Ink Density</th>
    <th>Centroid X</th>
    <th>Centroid Y</th>
    <th>Largest Component</th>
    <th>Horizontal Projection</th>
    <th>Vertical Projection</th>
    <th>Lower Ink</th>
    <th>Main Feature</th>
</tr>

{feature_rows}

</table>


<h2>
Automatic Feature Explanations
</h2>

{cards}


<div class="note">

<b>Interpretation note:</b>

The detected words are statistically unusual relative to the
10,000-word IAM reference sample. These results identify unusual
handwriting characteristics for further analysis; they do not
establish a clinical diagnosis of dysgraphia.

</div>


</body>

</html>
"""


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_HTML,
    "w",
    encoding="utf-8"
) as file:

    file.write(html_document)


print("=" * 60)
print("WORD ANOMALY REPORT CREATED")
print("=" * 60)

print()
print("Target: p06-052")
print("Words analyzed:", len(anomaly_df))
print("Reference words: 10000")
print()
print("Report:")
print(OUTPUT_HTML)