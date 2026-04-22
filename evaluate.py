import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RoadPondingDataset, get_transforms
from utils import MetricTracker, build_model, load_checkpoint_payload, load_model_weights, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AGSENet Classification Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights")
    parser.add_argument("--split", type=str, choices=["val", "test", "train"], default="val", help="Which split to evaluate")
    parser.add_argument("--export-gradcam", action="store_true", help="Delegate Grad-CAM export to visualize.py")
    parser.add_argument("--gradcam-target", type=str, choices=["predicted", "true"], default="predicted", help="Target class for Grad-CAM")
    return parser.parse_args()


def plot_tsne(features, targets, preds, class_names, out_dir):
    if len(features) < 2:
        return

    perplexity = min(30, max(5, len(features) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    correct_idx = targets == preds
    incorrect_idx = targets != preds
    palette = sns.color_palette("husl", len(class_names))

    for i, class_name in enumerate(class_names):
        class_correct = correct_idx & (targets == i)
        class_incorrect = incorrect_idx & (targets == i)

        if np.any(class_correct):
            plt.scatter(
                embedded[class_correct, 0],
                embedded[class_correct, 1],
                label=f"{class_name} (Correct)",
                color=palette[i],
                alpha=0.7,
                marker="o",
                s=50,
            )
        if np.any(class_incorrect):
            plt.scatter(
                embedded[class_incorrect, 0],
                embedded[class_incorrect, 1],
                label=f"{class_name} (Incorrect)",
                color=palette[i],
                alpha=1.0,
                marker="X",
                s=100,
                edgecolor="black",
            )

    plt.title("t-SNE Clustering")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_clusters_tsne.png"), dpi=300)
    plt.close()


@torch.no_grad()
def evaluate(model, dataloader, device, class_names):
    model.eval()
    tracker = MetricTracker(class_names=class_names)

    rows = []
    pooled_features = []
    for inputs, targets, paths in tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs, aux = model(inputs, return_aux=True)

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        tracker.update(preds, targets, probs)
        pooled_features.append(
            aux.get("enhanced_pooled_features", aux["pooled_features"]).detach().cpu().numpy()
        )
        visual_probs = torch.softmax(aux["visual_logits"], dim=1)
        text_probs = torch.softmax(aux["description_logits"], dim=1)

        topk_values, topk_indices = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
        for idx in range(inputs.shape[0]):
            row = {
                "path": paths[idx],
                "true_label": int(targets[idx].item()),
                "true_class": class_names[int(targets[idx].item())],
                "pred_label": int(preds[idx].item()),
                "pred_class": class_names[int(preds[idx].item())],
                "confidence": float(probs[idx, preds[idx]].item()),
                "correct": bool(preds[idx].item() == targets[idx].item()),
                "visual_pred": class_names[int(torch.argmax(visual_probs[idx]).item())],
                "text_pred": class_names[int(torch.argmax(text_probs[idx]).item())],
                "fusion_gate": float(aux["fusion_gate"][idx].item()) if "fusion_gate" in aux else None,
            }
            for class_idx, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probs[idx, class_idx].item())
            for rank, (value, cls_idx) in enumerate(zip(topk_values[idx], topk_indices[idx]), start=1):
                row[f"top_{rank}_class"] = class_names[int(cls_idx.item())]
                row[f"top_{rank}_prob"] = float(value.item())
            rows.append(row)

    return tracker, np.concatenate(pooled_features, axis=0), pd.DataFrame(rows)


def save_confusion_matrices(tracker, class_names, out_dir, split):
    cm = tracker.confusion_matrix()
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(out_dir, f"confusion_matrix_{split}.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({split})")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{split}.png"), dpi=300)
    plt.close()

    normalized = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    norm_df = pd.DataFrame(normalized, index=class_names, columns=class_names)
    norm_df.to_csv(os.path.join(out_dir, f"confusion_matrix_{split}_normalized.csv"))

    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_df, annot=True, fmt=".2f", cmap="mako")
    plt.title(f"Normalized Confusion Matrix ({split})")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{split}_normalized.png"), dpi=300)
    plt.close()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path, device)
    runtime_config = checkpoint_payload.get("config", config)
    class_names = checkpoint_payload.get("class_names")

    transform = get_transforms(runtime_config["image_size"], split=args.split)
    dataset = RoadPondingDataset(
        runtime_config["data_path"],
        transform=transform,
        split=args.split,
        merge_classes=runtime_config.get("merge_classes", False),
    )
    if class_names is None:
        class_names = dataset.classes

    dataloader = DataLoader(
        dataset,
        batch_size=runtime_config["batch_size"],
        shuffle=False,
        num_workers=runtime_config.get("num_workers", 4),
        pin_memory=True,
    )

    model, _ = build_model(config, class_names, device, checkpoint_payload=checkpoint_payload)
    missing, unexpected = load_model_weights(model, checkpoint_payload, strict=True)
    if missing or unexpected:
        print(f"Checkpoint load warnings | missing={missing} | unexpected={unexpected}")

    tracker, features, predictions_df = evaluate(model, dataloader, device, class_names)
    metrics = tracker.compute()

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / f"predictions_{args.split}.csv"
    predictions_df.to_csv(predictions_path, index=False)

    report_path = out_dir / f"classification_report_{args.split}.txt"
    report_path.write_text(tracker.classification_report(), encoding="utf-8")

    metrics_payload = {"split": args.split}
    metrics_payload.update({k: float(v) for k, v in metrics.items()})
    metrics_path = out_dir / f"evaluation_metrics_{args.split}.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    save_confusion_matrices(tracker, class_names, out_dir, args.split)
    plot_tsne(features, np.array(tracker.all_targets), np.array(tracker.all_preds), class_names, out_dir)

    print("\n--- Evaluation Results ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("\nClassification Report:")
    print(tracker.classification_report())
    print(f"\nSaved predictions to {predictions_path}")
    print(f"Saved report to {report_path}")

    if args.export_gradcam:
        subprocess.run(
            [
                "python",
                "visualize.py",
                "--config",
                args.config,
                "--split",
                args.split,
                "--gradcam-target",
                args.gradcam_target,
                "--export-encoder-gradcam",
                "--export-decoder-gradcam",
                "--best-checkpoint",
                str(checkpoint_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
