import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Training vs Validation Metrics")
    parser.add_argument('--log-file', type=str, default='outputs/metrics.jsonl', help='Path to metrics.jsonl log file generated during training')
    parser.add_argument('--out-dir', type=str, default='outputs/plots', help='Directory to save the plots')
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.log_file):
        print(f"Error: {args.log_file} does not exist. Please train the model first.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    
    data = []
    with open(args.log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    df = pd.DataFrame(data)
    
    # Filter for standard metrics that are typically compared
    keys = ['loss', 'acc', 'macro_f1', 'macro_precision', 'macro_recall', 'bal_acc', 'weighted_f1']
    
    for key in keys:
        if f"train_{key}" in df.columns and f"val_{key}" in df.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df, x='epoch', y=f"train_{key}", label=f"Train {key}", marker='o')
            sns.lineplot(data=df, x='epoch', y=f"val_{key}", label=f"Val {key}", marker='s')
            plt.title(f"Train vs Val {key.upper()}", fontsize=16)
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel(key.capitalize().replace("_", " "), fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"plot_train_val_{key}.png"), dpi=300)
            plt.close()
            print(f"Saved plot for {key}.")

    print(f"All metric plots saved securely to {args.out_dir}.")

if __name__ == "__main__":
    main()
