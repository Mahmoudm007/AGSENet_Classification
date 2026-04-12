import os
import sys
import argparse
import subprocess

def run_step(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with return code {result.returncode}. Aborting pipeline.")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Pipeline: EDA -> Architect -> Training -> Evaluation -> Visualization")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()

    python_exe = sys.executable
    best_ckpt = "outputs/checkpoints/best_model.pth"

    print("=" * 60)
    print("STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    # Using python -m format to ensure local imports in the scripts resolve properly
    run_step([python_exe, "eda/run_eda.py"])

    print("\n" + "=" * 60)
    print("STAGE 2: MODEL ARCHITECTURE DIAGRAM GENERATION")
    print("=" * 60)
    run_step([python_exe, "utils/model_diagram.py"])

    print("\n" + "=" * 60)
    print("STAGE 3: TRAINING")
    print("=" * 60)
    # Note: train.py also acts as a full driver invoking evaluation and visualization automatically now.
    run_step([python_exe, "train.py", "--config", args.config])

    print("\n" + "=" * 60)
    print("STAGE 4: EVALUATION ON TRAIN SET (CONFUSION MATRIX)")
    print("=" * 60)
    run_step([
        python_exe, "evaluate.py",
        "--checkpoint", best_ckpt,
        "--split", "train",
        "--config", args.config
    ])

    print("\n" + "=" * 60)
    print("STAGE 5: MANUAL EVALUATION OVERRIDE")
    print("=" * 60)
    run_step([
        python_exe, "evaluate.py",
        "--checkpoint", best_ckpt,
        "--split", "val",
        "--config", args.config,
        "--export-gradcam"
    ])

    print("\n" + "=" * 60)
    print("STAGE 6: MANUAL VISUALIZATION RE-GENERATION")
    print("=" * 60)
    run_step([
        python_exe, "visualize.py",
        "--config", args.config,
        "--split", "val",
        "--best-checkpoint", best_ckpt,
        "--export-image-panels",
        "--export-encoder-gradcam",
        "--export-decoder-gradcam",
        "--export-roc",
        "--export-recall-confidence"
    ])

    print("\n" + "=" * 60)
    print("STAGE 7: PLOT TRAINING METRICS")
    print("=" * 60)
    run_step([
        python_exe, "utils/plot_metrics.py",
        "--log-file", "outputs/metrics.jsonl",
        "--out-dir", "outputs/plots"
    ])

    print("\n" + "=" * 60)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"Results are saved in:")
    print(f" - outputs/checkpoints/")
    print(f" - outputs/visualizations/")
    print(f" - outputs/eda/")
    print(f" - outputs/model_architecture_diagram.html")
    print(f" - outputs/plots/")
    print("=" * 60)

if __name__ == "__main__":
    main()
