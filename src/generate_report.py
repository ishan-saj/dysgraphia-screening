import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def run_script(script_name, args=None):
    """Run another Python script and stop if it fails."""
    args = args or []

    script_path = SRC_DIR / script_name

    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print("=" * 70)

    command = [sys.executable, str(script_path)] + args

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed.")
        sys.exit(result.returncode)

    print(f"\nDONE: {script_name}")


def main():

    parser = argparse.ArgumentParser(
        description="Generate the complete handwriting explainability report."
    )

    parser.add_argument(
        "--sample",
        required=True,
        help="Sample ID, e.g. p06-052"
    )

    args = parser.parse_args()
    sample_id = args.sample

    print("\n" + "=" * 70)
    print("Dysgraphia Handwriting Explainability Report Generator")
    print("=" * 70)
    print(f"Sample: {sample_id}")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Word-level anomaly analysis
    # ---------------------------------------------------------
    word_anomaly_file = (
        METADATA_DIR / f"{sample_id}_word_anomaly.csv"
    )

    if word_anomaly_file.exists():
        print("\n[1/5] Word anomaly CSV already exists.")
        print("Skipping expensive word anomaly calculation.")
    else:
        run_script(
            "word_anomaly.py"
        )

    # ---------------------------------------------------------
    # 2. Word anomaly visualization
    # ---------------------------------------------------------
    run_script(
        "visualize_word_anomalies.py",
        ["--sample_id", sample_id]
    )

    # ---------------------------------------------------------
    # 3. Word explanations
    # ---------------------------------------------------------
    run_script(
        "explain_word_anomalies.py",
        ["--sample_id", sample_id]
    )

    # ---------------------------------------------------------
    # 4. Word Grad-CAM
    # ---------------------------------------------------------
    run_script(
        "word_gradcam.py",
        ["--sample_id", sample_id]
    )

    # ---------------------------------------------------------
    # 5. Final HTML report
    # ---------------------------------------------------------
    run_script(
        "create_final_report.py",
        ["--sample", sample_id]
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    report_file = (
        METADATA_DIR
        / "final_reports"
        / f"{sample_id}_final_report.html"
    )

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)

    if report_file.exists():
        print("\nFINAL REPORT:")
        print(report_file)
    else:
        print("\nWARNING:")
        print("The report command completed, but the expected HTML file")
        print("was not found at:")
        print(report_file)

    print("\n")


if __name__ == "__main__":
    main()