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
    parser = argparse.ArgumentParser(description="Run End-to-End Pipeline: Training -> Evaluation -> Visualization")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()

    python_exe = sys.executable
    best_ckpt = "outputs/checkpoints/best_model.pth"

    print("=" * 60)
    print("STAGE 1: TRAINING")
    print("=" * 60)
    run_step([python_exe, "train.py", "--config", args.config])

    print("\n" + "=" * 60)
    print("STAGE 2: EVALUATION (Standard Metrics)")
    print("=" * 60)
    run_step([
        python_exe, "evaluate.py",
        "--checkpoint", best_ckpt,
        "--split", "val",
        "--config", args.config
    ])

    print("\n" + "=" * 60)
    print("STAGE 3: PAPER-STYLE VISUALIZATIONS & GRAD-CAM")
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
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"Results are saved in:")
    print(f" - outputs/checkpoints/")
    print(f" - outputs/visualizations/")
    print("=" * 60)

if __name__ == "__main__":
    main()
