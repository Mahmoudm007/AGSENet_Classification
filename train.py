import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.dataset import RoadPondingDataset
from data.transforms import get_inverse_transform, get_transforms
from utils import (
    AverageMeter,
    CSVLogger,
    MetricTracker,
    append_parameter_reports,
    build_model,
    compute_multimodal_batch_stats,
    compute_class_weights,
    description_aux_enabled,
    get_display_names,
    get_loss_function,
    make_fixed_sample_batch,
    prototype_separation_loss,
    save_mix_snapshot,
    save_post_training_multimodal_analysis,
    save_text_feature_overview,
    save_training_dynamics_plots,
    set_seed,
    supervised_contrastive_loss,
    symmetric_kl_divergence,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train AGSENet Classification Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_sampler(dataset, config) -> Optional[WeightedRandomSampler]:
    if not config.get("use_weighted_sampler", False):
        return None

    class_counts = dataset.get_class_frequencies()
    sampler_mode = config.get("sampler_weight_mode", config.get("class_weight_mode", "effective_num"))
    class_weights = compute_class_weights(
        class_counts,
        mode=sampler_mode,
        beta=float(config.get("sampler_weight_beta", config.get("class_weight_beta", 0.9999))),
    )
    sample_weights = [float(class_weights[label].item()) for _, label in dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(f"Using WeightedRandomSampler with mode '{sampler_mode}'.")
    return sampler


def build_scheduler(optimizer, config):
    scheduler_type = config.get("scheduler_type", "cosine")
    epochs = int(config["epochs"])
    min_learning_rate = float(config.get("min_learning_rate", 1e-6))

    if scheduler_type == "warmup_cosine":
        warmup_epochs = max(1, int(config.get("warmup_epochs", 5)))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(config.get("warmup_start_factor", 0.1)),
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=min_learning_rate,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_learning_rate,
        )

    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config.get("step_scheduler_step_size", 20)),
        gamma=float(config.get("step_scheduler_gamma", 0.1)),
    )


def compute_losses(config, criterion, logits, aux_outputs, targets):
    total_loss = criterion(logits, targets)
    loss_items = {
        "combined_cls_loss": float(total_loss.detach().item()),
        "visual_cls_loss": 0.0,
        "description_cls_loss": 0.0,
        "enhanced_visual_loss": 0.0,
        "align_loss": 0.0,
        "consistency_loss": 0.0,
        "contrastive_loss": 0.0,
        "prototype_separation_loss": 0.0,
        "fusion_gate": 0.0,
    }

    if aux_outputs is None or "description_logits" not in aux_outputs:
        return total_loss, loss_items

    aux_cfg = config.get("description_aux", {})
    visual_weight = float(aux_cfg.get("visual_loss_weight", 0.5))
    description_weight = float(aux_cfg.get("description_loss_weight", 0.35))
    align_weight = float(aux_cfg.get("align_loss_weight", 0.15))
    enhanced_visual_weight = float(aux_cfg.get("enhanced_visual_loss_weight", 0.2))
    consistency_weight = float(aux_cfg.get("consistency_loss_weight", 0.1))
    contrastive_weight = float(aux_cfg.get("contrastive_loss_weight", 0.05))
    prototype_weight = float(aux_cfg.get("prototype_separation_weight", 0.05))

    visual_cls_loss = criterion(aux_outputs["visual_logits"], targets)
    description_cls_loss = criterion(aux_outputs["description_logits"], targets)
    enhanced_visual_logits = aux_outputs.get("enhanced_visual_logits")
    if enhanced_visual_logits is not None:
        enhanced_visual_loss = criterion(enhanced_visual_logits, targets)
    else:
        enhanced_visual_loss = visual_cls_loss.new_tensor(0.0)
    target_descriptions = aux_outputs["description_embedding_bank"].index_select(0, targets)
    align_loss = 1.0 - (aux_outputs["image_embedding"] * target_descriptions).sum(dim=1).mean()
    consistency_loss = symmetric_kl_divergence(
        aux_outputs["visual_logits"],
        aux_outputs["description_logits"],
        temperature=float(aux_cfg.get("consistency_temperature", 1.5)),
    )
    contrastive_loss = supervised_contrastive_loss(
        aux_outputs["image_embedding"],
        targets,
        temperature=float(aux_cfg.get("contrastive_temperature", 0.1)),
    )
    proto_loss = prototype_separation_loss(
        aux_outputs["description_embedding_bank"],
        margin=float(aux_cfg.get("prototype_margin", 0.2)),
    )

    total_loss = (
        total_loss
        + (visual_weight * visual_cls_loss)
        + (description_weight * description_cls_loss)
        + (enhanced_visual_weight * enhanced_visual_loss)
        + (align_weight * align_loss)
        + (consistency_weight * consistency_loss)
        + (contrastive_weight * contrastive_loss)
        + (prototype_weight * proto_loss)
    )
    loss_items.update(
        {
            "visual_cls_loss": float(visual_cls_loss.detach().item()),
            "description_cls_loss": float(description_cls_loss.detach().item()),
            "enhanced_visual_loss": float(enhanced_visual_loss.detach().item()),
            "align_loss": float(align_loss.detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "contrastive_loss": float(contrastive_loss.detach().item()),
            "prototype_separation_loss": float(proto_loss.detach().item()),
            "fusion_gate": float(aux_outputs.get("fusion_gate", torch.tensor(0.0, device=logits.device)).mean().detach().item()),
        }
    )
    return total_loss, loss_items


def _autocast_context(device):
    return torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda"))


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, metric_tracker, config):
    model.train()
    metric_tracker.reset()

    losses = AverageMeter()
    combined_losses = AverageMeter()
    visual_losses = AverageMeter()
    description_losses = AverageMeter()
    enhanced_visual_losses = AverageMeter()
    alignment_losses = AverageMeter()
    consistency_losses = AverageMeter()
    contrastive_losses = AverageMeter()
    prototype_losses = AverageMeter()
    fusion_gate_meter = AverageMeter()
    aux_meters = {
        "visual_acc": AverageMeter(),
        "description_acc": AverageMeter(),
        "branch_agreement": AverageMeter(),
        "fused_visual_agreement": AverageMeter(),
        "fused_text_agreement": AverageMeter(),
        "enhancement_agreement": AverageMeter(),
        "base_visual_acc": AverageMeter(),
        "enhanced_visual_acc": AverageMeter(),
        "true_visual_prob": AverageMeter(),
        "true_description_prob": AverageMeter(),
        "true_fused_prob": AverageMeter(),
        "true_text_similarity": AverageMeter(),
        "top_text_similarity": AverageMeter(),
        "avg_fusion_gate": AverageMeter(),
        "gate_correct_gap": AverageMeter(),
    }
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))

    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets, _ in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device):
            if description_aux_enabled(config):
                outputs, aux_outputs = model(inputs, return_aux=True)
            else:
                outputs = model(inputs)
                aux_outputs = None
            loss, loss_items = compute_losses(config, criterion, outputs, aux_outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), inputs.size(0))
        combined_losses.update(loss_items["combined_cls_loss"], inputs.size(0))
        visual_losses.update(loss_items["visual_cls_loss"], inputs.size(0))
        description_losses.update(loss_items["description_cls_loss"], inputs.size(0))
        enhanced_visual_losses.update(loss_items["enhanced_visual_loss"], inputs.size(0))
        alignment_losses.update(loss_items["align_loss"], inputs.size(0))
        consistency_losses.update(loss_items["consistency_loss"], inputs.size(0))
        contrastive_losses.update(loss_items["contrastive_loss"], inputs.size(0))
        prototype_losses.update(loss_items["prototype_separation_loss"], inputs.size(0))
        fusion_gate_meter.update(loss_items["fusion_gate"], inputs.size(0))
        batch_stats = compute_multimodal_batch_stats(outputs, aux_outputs, targets)
        for key, value in batch_stats.items():
            if key in aux_meters:
                aux_meters[key].update(value, inputs.size(0))

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        metric_tracker.update(preds, targets, probs)
        pbar.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "align": f"{alignment_losses.avg:.4f}",
            }
        )

    metrics = metric_tracker.compute()
    metrics["loss"] = losses.avg
    metrics["combined_cls_loss"] = combined_losses.avg
    metrics["visual_cls_loss"] = visual_losses.avg
    metrics["description_cls_loss"] = description_losses.avg
    metrics["enhanced_visual_loss"] = enhanced_visual_losses.avg
    metrics["align_loss"] = alignment_losses.avg
    metrics["consistency_loss"] = consistency_losses.avg
    metrics["contrastive_loss"] = contrastive_losses.avg
    metrics["prototype_separation_loss"] = prototype_losses.avg
    metrics["fusion_gate"] = fusion_gate_meter.avg
    for key, meter in aux_meters.items():
        metrics[key] = meter.avg
    return metrics


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, metric_tracker, epoch, out_dir, config):
    model.eval()
    metric_tracker.reset()

    losses = AverageMeter()
    combined_losses = AverageMeter()
    visual_losses = AverageMeter()
    description_losses = AverageMeter()
    enhanced_visual_losses = AverageMeter()
    alignment_losses = AverageMeter()
    consistency_losses = AverageMeter()
    contrastive_losses = AverageMeter()
    prototype_losses = AverageMeter()
    fusion_gate_meter = AverageMeter()
    aux_meters = {
        "visual_acc": AverageMeter(),
        "description_acc": AverageMeter(),
        "branch_agreement": AverageMeter(),
        "fused_visual_agreement": AverageMeter(),
        "fused_text_agreement": AverageMeter(),
        "enhancement_agreement": AverageMeter(),
        "base_visual_acc": AverageMeter(),
        "enhanced_visual_acc": AverageMeter(),
        "true_visual_prob": AverageMeter(),
        "true_description_prob": AverageMeter(),
        "true_fused_prob": AverageMeter(),
        "true_text_similarity": AverageMeter(),
        "top_text_similarity": AverageMeter(),
        "avg_fusion_gate": AverageMeter(),
        "gate_correct_gap": AverageMeter(),
    }
    sample_losses = []

    pbar = tqdm(dataloader, desc="Validation")
    for inputs, targets, paths in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_context(device):
            if description_aux_enabled(config):
                outputs, aux_outputs = model(inputs, return_aux=True)
            else:
                outputs = model(inputs)
                aux_outputs = None
            loss, loss_items = compute_losses(config, criterion, outputs, aux_outputs, targets)
            per_sample_loss = F.cross_entropy(outputs, targets, reduction="none")

        losses.update(loss.item(), inputs.size(0))
        combined_losses.update(loss_items["combined_cls_loss"], inputs.size(0))
        visual_losses.update(loss_items["visual_cls_loss"], inputs.size(0))
        description_losses.update(loss_items["description_cls_loss"], inputs.size(0))
        enhanced_visual_losses.update(loss_items["enhanced_visual_loss"], inputs.size(0))
        alignment_losses.update(loss_items["align_loss"], inputs.size(0))
        consistency_losses.update(loss_items["consistency_loss"], inputs.size(0))
        contrastive_losses.update(loss_items["contrastive_loss"], inputs.size(0))
        prototype_losses.update(loss_items["prototype_separation_loss"], inputs.size(0))
        fusion_gate_meter.update(loss_items["fusion_gate"], inputs.size(0))
        batch_stats = compute_multimodal_batch_stats(outputs, aux_outputs, targets)
        for key, value in batch_stats.items():
            if key in aux_meters:
                aux_meters[key].update(value, inputs.size(0))

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        metric_tracker.update(preds, targets, probs)
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})

        for b_idx in range(inputs.size(0)):
            sample_losses.append(
                {
                    "path": paths[b_idx],
                    "loss": float(per_sample_loss[b_idx].item()),
                    "pred": int(preds[b_idx].item()),
                    "true": int(targets[b_idx].item()),
                    "prob": float(probs[b_idx, preds[b_idx]].item()),
                }
            )

    sample_losses.sort(key=lambda item: item["loss"], reverse=True)
    hl_dir = Path(out_dir) / "high_loss_samples" / f"epoch_{epoch}"
    hl_dir.mkdir(parents=True, exist_ok=True)

    top_losses = sample_losses[:20]
    for rank, row in enumerate(top_losses, start=1):
        ext = Path(row["path"]).suffix
        new_name = f"Rank{rank}_Loss{row['loss']:.3f}_P{row['pred']}_T{row['true']}{ext}"
        try:
            shutil.copy(row["path"], hl_dir / new_name)
        except Exception:
            pass
    pd.DataFrame(top_losses).to_csv(hl_dir / "top_losses.csv", index=False)

    metrics = metric_tracker.compute()
    metrics["loss"] = losses.avg
    metrics["combined_cls_loss"] = combined_losses.avg
    metrics["visual_cls_loss"] = visual_losses.avg
    metrics["description_cls_loss"] = description_losses.avg
    metrics["enhanced_visual_loss"] = enhanced_visual_losses.avg
    metrics["align_loss"] = alignment_losses.avg
    metrics["consistency_loss"] = consistency_losses.avg
    metrics["contrastive_loss"] = contrastive_losses.avg
    metrics["prototype_separation_loss"] = prototype_losses.avg
    metrics["fusion_gate"] = fusion_gate_meter.avg
    for key, meter in aux_meters.items():
        metrics[key] = meter.avg
    return metrics


def save_description_metadata(output_dir: str, metadata: Dict, class_names):
    payload = {
        "class_names": class_names,
        "display_names": get_display_names(class_names),
        "description_aux": metadata,
    }
    meta_path = Path(output_dir) / "class_description_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta_path


def build_checkpoint_payload(model, config, epoch, best_val_loss, class_names, text_metadata):
    return {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "config": config,
        "text_metadata": text_metadata,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config["output_dir"], exist_ok=True)
    for generated_file in ["training_log.csv", "metrics.jsonl"]:
        generated_path = Path(config["output_dir"]) / generated_file
        if generated_path.exists():
            generated_path.unlink()

    train_transform = get_transforms(config["image_size"], split="train")
    val_transform = get_transforms(config["image_size"], split="val")

    train_dataset = RoadPondingDataset(
        config["data_path"],
        transform=train_transform,
        split="train",
        merge_classes=config.get("merge_classes", False),
    )
    val_dataset = RoadPondingDataset(
        config["data_path"],
        transform=val_transform,
        split="val",
        merge_classes=config.get("merge_classes", False),
    )
    if len(train_dataset.classes) != int(config["num_classes"]):
        raise ValueError(
            f"Config num_classes={config['num_classes']} but dataset exposes "
            f"{len(train_dataset.classes)} classes: {train_dataset.classes}"
        )

    sampler = build_train_sampler(train_dataset, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    class_frequencies = train_dataset.get_class_frequencies()
    criterion = get_loss_function(config, class_frequencies=class_frequencies).to(device)

    model, text_metadata = build_model(config, train_dataset.classes, device)
    save_description_metadata(config["output_dir"], text_metadata, train_dataset.classes)
    inverse_transform = get_inverse_transform()
    fixed_batch = None
    aux_cfg = config.get("description_aux", {})
    if description_aux_enabled(config):
        save_text_feature_overview(
            config["output_dir"],
            train_dataset.classes,
            text_metadata,
            model.description_embeddings.detach().cpu(),
        )
        fixed_batch = make_fixed_sample_batch(
            val_loader,
            max_samples=int(aux_cfg.get("snapshot_num_images", 12)),
        )
        save_mix_snapshot(
            model,
            fixed_batch,
            device,
            train_dataset.classes,
            inverse_transform,
            Path(config["output_dir"]) / "multimodal_analysis" / "before_training",
            "epoch_000_before_training",
        )

    overview_path = Path(config["output_dir"]) / "parameter_overview.csv"
    breakdown_path = Path(config["output_dir"]) / "parameter_breakdown.csv"
    if overview_path.exists():
        overview_path.unlink()
    if breakdown_path.exists():
        breakdown_path.unlink()

    param_summary = append_parameter_reports(model, config["output_dir"], epoch=0)
    print(f"Parameter reports: {param_summary[0]}, {param_summary[1]}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(
            float(config.get("adam_beta1", 0.9)),
            float(config.get("adam_beta2", 0.999)),
        ),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, config)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    metric_tracker = MetricTracker(class_names=train_dataset.classes)
    logger = CSVLogger(config["output_dir"])
    history_rows = []

    best_val_loss = float("inf")
    checkpoints_dir = Path(config["output_dir"]) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        append_parameter_reports(model, config["output_dir"], epoch=epoch)

        print(f"\n[{epoch}/{config['epochs']}] Training...")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, metric_tracker, config)
        logger.print_metrics(epoch, train_metrics, is_val=False)

        if epoch % config.get("validate_every_n_epochs", 1) == 0:
            val_metrics = validate_epoch(
                model,
                val_loader,
                criterion,
                device,
                metric_tracker,
                epoch,
                config["output_dir"],
                config,
            )
            logger.print_metrics(epoch, val_metrics, is_val=True)

            combined_metrics = {"lr": optimizer.param_groups[0]["lr"]}
            combined_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
            combined_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            logger.log(epoch, combined_metrics)
            history_rows.append({"epoch": epoch, **combined_metrics})
            history_df = pd.DataFrame(history_rows)
            history_df.to_csv(Path(config["output_dir"]) / "multimodal_history.csv", index=False)
            save_training_dynamics_plots(history_df, config["output_dir"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_path = checkpoints_dir / "best_model.pth"
                torch.save(
                    build_checkpoint_payload(
                        model=model,
                        config=config,
                        epoch=epoch,
                        best_val_loss=best_val_loss,
                        class_names=train_dataset.classes,
                        text_metadata=text_metadata,
                    ),
                    best_path,
                )
                print(f"Saved new best model with validation loss {best_val_loss:.4f}")

            if description_aux_enabled(config):
                snapshot_every = int(aux_cfg.get("snapshot_every_n_epochs", 10))
                if snapshot_every > 0 and (epoch == 1 or epoch % snapshot_every == 0 or epoch == config["epochs"]):
                    save_mix_snapshot(
                        model,
                        fixed_batch,
                        device,
                        train_dataset.classes,
                        inverse_transform,
                        Path(config["output_dir"]) / "multimodal_analysis" / "during_training",
                        f"epoch_{epoch:03d}",
                    )

        scheduler.step()

    final_path = checkpoints_dir / "final_model.pth"
    torch.save(
        build_checkpoint_payload(
            model=model,
            config=config,
            epoch=config["epochs"],
            best_val_loss=best_val_loss,
            class_names=train_dataset.classes,
            text_metadata=text_metadata,
        ),
        final_path,
    )
    print("Training completed.")

    print("\nRunning evaluation and visualization on best checkpoint...")
    best_path = checkpoints_dir / "best_model.pth"
    if best_path.exists():
        try:
            best_payload = torch.load(best_path, map_location=device)
            model.load_state_dict(best_payload["state_dict"])
            model.to(device)
            model.eval()
            if description_aux_enabled(config):
                save_post_training_multimodal_analysis(
                    model,
                    val_loader,
                    device,
                    train_dataset.classes,
                    config["output_dir"],
                    split="val",
                )
            subprocess.run(
                ["python", "evaluate.py", "--config", args.config, "--checkpoint", str(best_path), "--split", "val"],
                check=True,
            )
            subprocess.run(
                [
                    "python",
                    "visualize.py",
                    "--config",
                    args.config,
                    "--split",
                    "val",
                    "--export-image-panels",
                    "--export-encoder-gradcam",
                    "--export-decoder-gradcam",
                    "--export-roc",
                    "--export-pr",
                    "--export-topk",
                    "--export-calibration",
                    "--best-checkpoint",
                    str(best_path),
                ],
                check=True,
            )
        except Exception as exc:
            print(f"Automatic post-training evaluation failed: {exc}")


if __name__ == "__main__":
    main()
