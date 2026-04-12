import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import RoadPondingDataset
from data.transforms import get_inverse_transform, get_transforms
from utils import (
    build_model,
    get_class_display_name,
    get_display_names,
    load_checkpoint_payload,
    load_model_weights,
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def parse_args():
    parser = argparse.ArgumentParser(description="AGSENet Classification visualizer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Dataset split to visualize")
    parser.add_argument("--export-image-panels", action="store_true", help="Export validation image figures with titles")
    parser.add_argument("--export-encoder-gradcam", action="store_true", help="Export encoder stage Grad-CAM figures")
    parser.add_argument("--export-decoder-gradcam", action="store_true", help="Export decoder stage Grad-CAM figures")
    parser.add_argument("--export-roc", action="store_true", help="Export multi-class ROC curve")
    parser.add_argument("--export-pr", action="store_true", help="Export multi-class precision-recall curve")
    parser.add_argument("--export-topk", action="store_true", help="Export per-image top-k probability panels")
    parser.add_argument("--export-calibration", action="store_true", help="Export confidence calibration and normalized confusion plots")
    parser.add_argument("--gradcam-target", type=str, default="predicted", choices=["predicted", "true"], help="Target class for Grad-CAM")
    parser.add_argument("--max-images", type=int, default=None, help="Limit the number of images to process")
    parser.add_argument("--best-checkpoint", type=str, default="auto", help="Path to weights, or auto to find best_model.pth")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(text: str) -> str:
    for token in ["<", ">", ":", "\"", "/", "\\", "|", "?", "*"]:
        text = text.replace(token, "_")
    return text.strip()


def create_output_dirs(base_dir, split, class_names):
    base_dir = Path(base_dir)
    dirs = {
        "plots": base_dir / "visualizations" / split,
        "gradcam": base_dir / "gradcam" / split,
        "actual_images": base_dir / "actual_images" / split,
        "topk": base_dir / "visualizations" / split / "top_k",
        "prediction_panels": base_dir / "visualizations" / split / "prediction_panels",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        display_name = get_class_display_name(class_name)
        class_gradcam = dirs["gradcam"] / display_name
        class_actual = dirs["actual_images"] / display_name
        class_topk = dirs["topk"] / display_name
        class_prediction = dirs["prediction_panels"] / display_name
        for folder in [class_gradcam, class_actual, class_topk, class_prediction]:
            folder.mkdir(parents=True, exist_ok=True)

        for idx in range(1, 7):
            (class_gradcam / f"Encoder {idx}").mkdir(exist_ok=True)
        for idx in range(1, 5):
            (class_gradcam / f"Decoder {idx}").mkdir(exist_ok=True)
        (class_gradcam / "Combined_Encoders").mkdir(exist_ok=True)
        (class_gradcam / "Combined_Decoder").mkdir(exist_ok=True)

    return dirs


def generate_figure_panel(original_img, heatmaps, titles, combined_save_path):
    n_panels = 1 + len(heatmaps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    for i, (heatmap, title) in enumerate(zip(heatmaps, titles), start=1):
        axes[i].imshow(heatmap)
        axes[i].set_title(title, fontsize=13)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(combined_save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_probs, class_names, save_dir):
    n_classes = len(class_names)
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=np.float32)
    for idx, label in enumerate(y_true):
        y_true_bin[idx, label] = 1.0

    fpr = {}
    tpr = {}
    roc_auc = {}
    rows = []

    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) < 2:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        rows.append({"class_name": class_names[i], "auc": float(roc_auc[i])})

    if not roc_auc:
        return

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    valid_keys = [key for key in range(n_classes) if key in fpr]
    all_fpr = np.unique(np.concatenate([fpr[key] for key in valid_keys]))
    mean_tpr = np.zeros_like(all_fpr)
    for key in valid_keys:
        mean_tpr += np.interp(all_fpr, fpr[key], tpr[key])
    mean_tpr /= max(len(valid_keys), 1)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC={roc_auc['micro']:.3f})", color="deeppink", linestyle=":", linewidth=3)
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC={roc_auc['macro']:.3f})", color="navy", linestyle=":", linewidth=3)

    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_classes))
    for idx, color in zip(valid_keys, colors):
        plt.plot(fpr[idx], tpr[idx], color=color, lw=2, label=f"{class_names[idx]} (AUC={roc_auc[idx]:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "roc_curve.png", dpi=300)
    plt.close()

    pd.DataFrame(rows + [{"class_name": "micro", "auc": float(roc_auc["micro"])}, {"class_name": "macro", "auc": float(roc_auc["macro"])}]).to_csv(
        save_dir / "roc_auc_scores.csv",
        index=False,
    )


def plot_pr_curve(y_true, y_probs, class_names, save_dir):
    n_classes = len(class_names)
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=np.float32)
    for idx, label in enumerate(y_true):
        y_true_bin[idx, label] = 1.0

    plt.figure(figsize=(10, 8))
    rows = []
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        if len(np.unique(y_true_bin[:, i])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        pr_auc = auc(recall, precision)
        rows.append({"class_name": class_names[i], "pr_auc": float(pr_auc)})
        plt.plot(recall, precision, color=color, lw=2, label=f"{class_names[i]} (AUC={pr_auc:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-class Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_dir / "precision_recall_curve.png", dpi=300)
    plt.close()

    if rows:
        pd.DataFrame(rows).to_csv(save_dir / "precision_recall_auc_scores.csv", index=False)


def plot_calibration(y_true, y_probs, save_dir):
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(confidences, bins, right=True)

    rows = []
    avg_conf = []
    avg_acc = []
    for idx in range(1, len(bins) + 1):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        conf_mean = float(confidences[mask].mean())
        acc_mean = float(correctness[mask].mean())
        avg_conf.append(conf_mean)
        avg_acc.append(acc_mean)
        rows.append({"bin": idx, "avg_confidence": conf_mean, "accuracy": acc_mean, "count": int(mask.sum())})

    if not rows:
        return

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(avg_conf, avg_acc, marker="o", linewidth=2, label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Confidence Calibration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "confidence_calibration.png", dpi=300)
    plt.close()

    pd.DataFrame(rows).to_csv(save_dir / "confidence_calibration.csv", index=False)


def plot_normalized_confusion(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    normalized = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    df = pd.DataFrame(normalized, index=class_names, columns=class_names)
    df.to_csv(save_dir / "normalized_confusion_matrix.csv")

    plt.figure(figsize=(10, 8))
    plt.imshow(normalized, cmap="magma", interpolation="nearest")
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_dir / "normalized_confusion_matrix.png", dpi=300)
    plt.close()


def save_prediction_panel(rgb_img, true_name, pred_name, confidence, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_img)
    color = "green" if true_name == pred_name else "red"
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4)
    ax.set_title(f"True: {true_name} | Pred: {pred_name} | Conf: {confidence:.2f}", fontsize=13, color=color, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_topk_panel(rgb_img, topk_names, topk_scores, true_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"True Class: {true_name}", fontsize=12)
    axes[0].axis("off")

    y_pos = np.arange(len(topk_names))
    axes[1].barh(y_pos, topk_scores, color=plt.cm.Blues(np.linspace(0.45, 0.9, len(topk_names))))
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(topk_names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Top-K Predictions", fontsize=12)
    for idx, score in enumerate(topk_scores):
        axes[1].text(min(score + 0.02, 0.98), idx, f"{score:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.best_checkpoint == "auto":
        checkpoint_path = Path(config["output_dir"]) / "checkpoints" / "best_model.pth"
    else:
        checkpoint_path = Path(args.best_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint_payload = load_checkpoint_payload(checkpoint_path, device)
    runtime_config = checkpoint_payload.get("config", config)
    class_names = checkpoint_payload.get("class_names")

    dataset = RoadPondingDataset(
        runtime_config["data_path"],
        transform=get_transforms(runtime_config["image_size"], split=args.split),
        split=args.split,
        merge_classes=runtime_config.get("merge_classes", False),
    )
    if class_names is None:
        class_names = dataset.classes
    display_names = get_display_names(class_names)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=runtime_config.get("num_workers", 4))
    inv_transform = get_inverse_transform()

    dirs = create_output_dirs(config["output_dir"], args.split, class_names)

    model, _ = build_model(config, class_names, device, checkpoint_payload=checkpoint_payload)
    missing, unexpected = load_model_weights(model, checkpoint_payload, strict=True)
    if missing or unexpected:
        print(f"Checkpoint load warnings | missing={missing} | unexpected={unexpected}")
    model.eval()

    encoder_layers = [model.en_1, model.en_2, model.en_3, model.en_4, model.en_5, model.en_6]
    decoder_layers = [model.proj_6, model.ssie_5, model.ssie_4, model.ssie_3]

    metadata = []
    y_true = []
    y_pred = []
    y_probs = []

    for idx, (inputs, targets, paths) in enumerate(tqdm(dataloader, desc=f"Processing {args.split}")):
        if args.max_images is not None and idx >= args.max_images:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[0].detach().cpu()

        pred_label = int(torch.argmax(probs).item())
        true_label = int(targets.item())
        pred_name = display_names[pred_label]
        true_name = display_names[true_label]
        confidence = float(probs[pred_label].item())
        source_name = safe_name(Path(paths[0]).stem)
        filename = safe_name(f"{source_name}_Predicted_{pred_name} and True_{true_name}.png")

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_probs.append(probs.numpy())

        pil_image = inv_transform(inputs[0])
        rgb_uint8 = np.array(pil_image)
        rgb_img = np.float32(rgb_uint8) / 255.0

        actual_image_path = dirs["actual_images"] / true_name / filename
        pil_image.save(actual_image_path)

        panel_path = dirs["prediction_panels"] / true_name / filename
        if args.export_image_panels:
            save_prediction_panel(rgb_img, true_name, pred_name, confidence, panel_path)

        topk_vals, topk_idx = torch.topk(probs, k=min(3, len(class_names)))
        topk_names = [display_names[int(class_idx)] for class_idx in topk_idx.tolist()]
        topk_scores = [float(value) for value in topk_vals.tolist()]
        topk_panel_path = dirs["topk"] / true_name / filename
        if args.export_topk:
            save_topk_panel(rgb_img, topk_names, topk_scores, true_name, topk_panel_path)

        cam_target = [ClassifierOutputTarget(true_label if args.gradcam_target == "true" else pred_label)]

        enc_combined_path = None
        if args.export_encoder_gradcam:
            encoder_heatmaps = []
            for stage_idx, layer in enumerate(encoder_layers, start=1):
                with GradCAM(model=model, target_layers=[layer]) as cam:
                    grayscale_cam = cam(input_tensor=inputs, targets=cam_target)[0]
                overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                encoder_heatmaps.append(overlay)
                stage_path = dirs["gradcam"] / true_name / f"Encoder {stage_idx}" / filename
                cv2.imwrite(str(stage_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            enc_combined_path = dirs["gradcam"] / true_name / "Combined_Encoders" / filename
            generate_figure_panel(rgb_uint8, encoder_heatmaps, [f"Encoder {i}" for i in range(1, 7)], enc_combined_path)

        dec_combined_path = None
        if args.export_decoder_gradcam:
            decoder_heatmaps = []
            for stage_idx, layer in enumerate(decoder_layers, start=1):
                with GradCAM(model=model, target_layers=[layer]) as cam:
                    grayscale_cam = cam(input_tensor=inputs, targets=cam_target)[0]
                overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                decoder_heatmaps.append(overlay)
                stage_path = dirs["gradcam"] / true_name / f"Decoder {stage_idx}" / filename
                cv2.imwrite(str(stage_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            dec_combined_path = dirs["gradcam"] / true_name / "Combined_Decoder" / filename
            generate_figure_panel(rgb_uint8, decoder_heatmaps, [f"Decoder {i}" for i in range(1, 5)], dec_combined_path)

        row = {
            "path": paths[0],
            "true_label": true_label,
            "true_class": true_name,
            "pred_label": pred_label,
            "pred_class": pred_name,
            "confidence": confidence,
            "actual_image_path": str(actual_image_path),
            "prediction_panel_path": str(panel_path) if args.export_image_panels else None,
            "topk_panel_path": str(topk_panel_path) if args.export_topk else None,
            "combined_encoder_path": str(enc_combined_path) if enc_combined_path else None,
            "combined_decoder_path": str(dec_combined_path) if dec_combined_path else None,
        }
        for class_idx, class_name in enumerate(display_names):
            row[f"prob_{class_name}"] = float(probs[class_idx].item())
        for rank, (name, score) in enumerate(zip(topk_names, topk_scores), start=1):
            row[f"top_{rank}_class"] = name
            row[f"top_{rank}_prob"] = score
        metadata.append(row)

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_probs_np = np.array(y_probs)

    if args.export_roc:
        plot_roc_curve(y_true_np, y_probs_np, display_names, dirs["plots"])
    if args.export_pr:
        plot_pr_curve(y_true_np, y_probs_np, display_names, dirs["plots"])
    if args.export_calibration:
        plot_calibration(y_true_np, y_probs_np, dirs["plots"])
        plot_normalized_confusion(y_true_np, y_pred_np, display_names, dirs["plots"])

    pd.DataFrame(metadata).to_csv(dirs["plots"] / f"metadata_{args.split}.csv", index=False)
    print(f"Completed. Outputs saved under {Path(config['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
