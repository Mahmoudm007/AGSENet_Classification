import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_colormaps
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA


def compute_multimodal_batch_stats(logits, aux_outputs, targets) -> Dict[str, float]:
    if aux_outputs is None or "description_logits" not in aux_outputs:
        return {}

    visual_probs = torch.softmax(aux_outputs["visual_logits"], dim=1)
    text_probs = torch.softmax(aux_outputs["description_logits"], dim=1)
    fused_probs = torch.softmax(logits, dim=1)

    visual_preds = torch.argmax(visual_probs, dim=1)
    text_preds = torch.argmax(text_probs, dim=1)
    fused_preds = torch.argmax(fused_probs, dim=1)

    row_idx = torch.arange(targets.size(0), device=targets.device)
    cosine_matrix = torch.matmul(aux_outputs["image_embedding"], aux_outputs["description_embedding_bank"].T)

    stats = {
        "visual_acc": float((visual_preds == targets).float().mean().item()),
        "description_acc": float((text_preds == targets).float().mean().item()),
        "branch_agreement": float((visual_preds == text_preds).float().mean().item()),
        "fused_visual_agreement": float((fused_preds == visual_preds).float().mean().item()),
        "fused_text_agreement": float((fused_preds == text_preds).float().mean().item()),
        "true_visual_prob": float(visual_probs[row_idx, targets].mean().item()),
        "true_description_prob": float(text_probs[row_idx, targets].mean().item()),
        "true_fused_prob": float(fused_probs[row_idx, targets].mean().item()),
        "true_text_similarity": float(cosine_matrix[row_idx, targets].mean().item()),
        "top_text_similarity": float(cosine_matrix.max(dim=1).values.mean().item()),
    }
    if "base_visual_logits" in aux_outputs:
        base_probs = torch.softmax(aux_outputs["base_visual_logits"], dim=1)
        base_preds = torch.argmax(base_probs, dim=1)
        stats["base_visual_acc"] = float((base_preds == targets).float().mean().item())
    if "enhanced_visual_logits" in aux_outputs:
        enhanced_probs = torch.softmax(aux_outputs["enhanced_visual_logits"], dim=1)
        enhanced_preds = torch.argmax(enhanced_probs, dim=1)
        stats["enhanced_visual_acc"] = float((enhanced_preds == targets).float().mean().item())
        if "base_visual_logits" in aux_outputs:
            stats["enhancement_agreement"] = float((enhanced_preds == base_preds).float().mean().item())
    if "fusion_gate" in aux_outputs:
        fusion_gate = aux_outputs["fusion_gate"].view(-1)
        stats["avg_fusion_gate"] = float(fusion_gate.mean().item())
        correct_mask = fused_preds == targets
        if torch.any(correct_mask) and torch.any(~correct_mask):
            stats["gate_correct_gap"] = float(
                fusion_gate[correct_mask].mean().item() - fusion_gate[~correct_mask].mean().item()
            )
        else:
            stats["gate_correct_gap"] = 0.0
    return stats


def _resolve_cmap(cmap: str) -> str:
    if cmap in mpl_colormaps:
        return cmap
    fallback_map = {
        "mako": "viridis",
        "rocket": "magma",
        "crest": "cividis",
        "flare": "plasma",
    }
    return fallback_map.get(cmap, "viridis")


def _save_heatmap(matrix, row_labels, col_labels, title, save_path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 1.3), max(4, len(row_labels) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap=_resolve_cmap(cmap))
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _truncate_path(path: str, max_len: int = 28) -> str:
    name = Path(path).stem
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def save_text_feature_overview(output_dir: str, class_names, text_metadata, description_embeddings):
    base_dir = Path(output_dir) / "multimodal_analysis" / "before_training"
    base_dir.mkdir(parents=True, exist_ok=True)

    embeddings = torch.as_tensor(description_embeddings, dtype=torch.float32)
    embeddings = F.normalize(embeddings, dim=1)
    similarity = (embeddings @ embeddings.T).cpu().numpy()
    _save_heatmap(similarity, class_names, class_names, "Description Prototype Cosine Similarity", base_dir / "description_similarity_heatmap.png")

    fig, axes = plt.subplots(len(class_names), 1, figsize=(12, max(10, len(class_names) * 2.8)))
    axes = np.atleast_1d(axes)
    for ax, class_name, terms in zip(axes, class_names, text_metadata.get("top_terms_per_text", [])):
        term_names = [item["term"] for item in terms][:10]
        weights = [item["weight"] for item in terms][:10]
        ax.barh(term_names[::-1], weights[::-1], color=plt.cm.cividis(np.linspace(0.25, 0.85, max(1, len(term_names)))))
        ax.set_title(f"Top TF-IDF Terms: {class_name}")
        ax.set_xlabel("TF-IDF Weight")
    fig.tight_layout()
    fig.savefig(base_dir / "top_tfidf_terms.png", dpi=220)
    plt.close(fig)

    if embeddings.shape[1] >= 2:
        coords = PCA(n_components=2).fit_transform(embeddings.cpu().numpy())
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(coords[:, 0], coords[:, 1], s=120, c=np.arange(len(class_names)), cmap="tab10")
        for idx, class_name in enumerate(class_names):
            ax.text(coords[idx, 0], coords[idx, 1], class_name, fontsize=11, ha="left", va="bottom")
        ax.set_title("Description Embedding PCA Projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(base_dir / "description_embedding_pca.png", dpi=220)
        plt.close(fig)

    metadata_path = base_dir / "text_feature_metadata.json"
    metadata_path.write_text(json.dumps(text_metadata, indent=2), encoding="utf-8")


@torch.no_grad()
def make_fixed_sample_batch(dataloader, max_samples: int = 12):
    inputs_list = []
    targets_list = []
    paths: List[str] = []

    for inputs, targets, batch_paths in dataloader:
        for idx in range(inputs.size(0)):
            inputs_list.append(inputs[idx : idx + 1].cpu())
            targets_list.append(targets[idx : idx + 1].cpu())
            paths.append(batch_paths[idx])
            if len(paths) >= max_samples:
                return {
                    "inputs": torch.cat(inputs_list, dim=0),
                    "targets": torch.cat(targets_list, dim=0),
                    "paths": paths,
                }

    if not paths:
        return None
    return {
        "inputs": torch.cat(inputs_list, dim=0),
        "targets": torch.cat(targets_list, dim=0),
        "paths": paths,
    }


@torch.no_grad()
def save_mix_snapshot(
    model,
    sample_batch,
    device,
    class_names,
    inverse_transform,
    save_dir,
    tag: str,
):
    if sample_batch is None:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    inputs = sample_batch["inputs"].to(device)
    targets = sample_batch["targets"].to(device)
    paths = sample_batch["paths"]

    model.eval()
    logits, aux = model(inputs, return_aux=True)
    visual_probs = torch.softmax(aux["visual_logits"], dim=1).cpu().numpy()
    text_probs = torch.softmax(aux["description_logits"], dim=1).cpu().numpy()
    fused_probs = torch.softmax(logits, dim=1).cpu().numpy()
    cosine_matrix = torch.matmul(aux["image_embedding"], aux["description_embedding_bank"].T).cpu().numpy()
    base_visual_probs = None
    enhanced_visual_probs = None
    fusion_gate = None
    if "base_visual_logits" in aux:
        base_visual_probs = torch.softmax(aux["base_visual_logits"], dim=1).cpu().numpy()
    if "enhanced_visual_logits" in aux:
        enhanced_visual_probs = torch.softmax(aux["enhanced_visual_logits"], dim=1).cpu().numpy()
    if "fusion_gate" in aux:
        fusion_gate = aux["fusion_gate"].detach().cpu().numpy()

    row_labels = [
        f"{idx:02d} | T:{class_names[int(target.item())]} | {_truncate_path(path)}"
        for idx, (target, path) in enumerate(zip(sample_batch["targets"], paths))
    ]

    panels = [
        (visual_probs, "Visual Branch Probabilities", class_names),
        (text_probs, "TF-IDF Text Branch Probabilities", class_names),
        (fused_probs, "Fused Probabilities", class_names),
        (cosine_matrix, "Image-to-Description Cosine Similarity", class_names),
    ]
    if base_visual_probs is not None:
        panels.append((base_visual_probs, "Base Visual Head Probabilities", class_names))
    if enhanced_visual_probs is not None:
        panels.append((enhanced_visual_probs, "Enhanced Visual Head Probabilities", class_names))
    if fusion_gate is not None:
        panels.append((fusion_gate, "Dynamic Fusion Gate", ["gate"]))

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, max(6, nrows * 4.4)))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (matrix, title, col_labels) in zip(axes.flat, panels):
        ax.axis("on")
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_dir / f"{tag}_mix_heatmaps.png", dpi=220)
    plt.close(fig)

    ncols = min(4, len(paths))
    nrows = int(np.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    fused_pred = fused_probs.argmax(axis=1)
    visual_pred = visual_probs.argmax(axis=1)
    text_pred = text_probs.argmax(axis=1)
    base_pred = base_visual_probs.argmax(axis=1) if base_visual_probs is not None else None
    enhanced_pred = enhanced_visual_probs.argmax(axis=1) if enhanced_visual_probs is not None else None
    for ax in axes.flat:
        ax.axis("off")
    for idx, ax in enumerate(axes.flat[: len(paths)]):
        image = inverse_transform(sample_batch["inputs"][idx])
        gate_text = f" G:{float(fusion_gate[idx, 0]):.2f}" if fusion_gate is not None else ""
        base_text = f" B:{class_names[int(base_pred[idx])]}" if base_pred is not None else ""
        enhanced_text = f" E:{class_names[int(enhanced_pred[idx])]}" if enhanced_pred is not None else ""
        ax.imshow(np.array(image))
        ax.set_title(
            f"T:{class_names[int(sample_batch['targets'][idx].item())]}\n"
            f"F:{class_names[int(fused_pred[idx])]} "
            f"V:{class_names[int(visual_pred[idx])]} "
            f"X:{class_names[int(text_pred[idx])]}\n"
            f"{base_text}{enhanced_text}{gate_text}".strip(),
            fontsize=10,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_dir / f"{tag}_sample_grid.png", dpi=220)
    plt.close(fig)

    if "scale_token_weights" in aux:
        token_labels = [
            "Avg f3", "Max f3", "Avg f4", "Max f4", "Avg f5", "Max f5",
            "Avg f6", "Max f6", "GeM f3", "GeM f4", "GeM f5", "GeM f6",
        ]
        token_matrix = aux["scale_token_weights"].detach().cpu().numpy()
        _save_heatmap(
            token_matrix,
            row_labels,
            token_labels,
            "Scale Token Weights",
            save_dir / f"{tag}_scale_token_weights.png",
            cmap="cividis",
        )

    rows = []
    for idx, path in enumerate(paths):
        true_label = int(sample_batch["targets"][idx].item())
        rows.append(
            {
                "path": path,
                "true_class": class_names[true_label],
                "fused_pred": class_names[int(fused_pred[idx])],
                "visual_pred": class_names[int(visual_pred[idx])],
                "text_pred": class_names[int(text_pred[idx])],
                "base_visual_pred": class_names[int(base_pred[idx])] if base_pred is not None else None,
                "enhanced_visual_pred": class_names[int(enhanced_pred[idx])] if enhanced_pred is not None else None,
                "true_visual_prob": float(visual_probs[idx, true_label]),
                "true_text_prob": float(text_probs[idx, true_label]),
                "true_fused_prob": float(fused_probs[idx, true_label]),
                "true_text_similarity": float(cosine_matrix[idx, true_label]),
                "fusion_gate": float(fusion_gate[idx, 0]) if fusion_gate is not None else None,
            }
        )
    pd.DataFrame(rows).to_csv(save_dir / f"{tag}_sample_metrics.csv", index=False)


def save_training_dynamics_plots(history_df: pd.DataFrame, output_dir: str):
    if history_df.empty:
        return

    base_dir = Path(output_dir) / "multimodal_analysis" / "during_training"
    base_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    curves = [
        ("align_loss", "Alignment Loss"),
        ("visual_acc", "Visual Branch Accuracy"),
        ("branch_agreement", "Branch Agreement"),
        ("enhanced_visual_acc", "Enhanced Visual Head Accuracy"),
        ("avg_fusion_gate", "Fusion Gate"),
        ("true_fused_prob", "True-Class Probability"),
    ]
    for ax, (metric, title) in zip(axes.flat, curves):
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        if train_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[train_col], label=train_col, marker="o")
        if val_col in history_df.columns:
            ax.plot(history_df["epoch"], history_df[val_col], label=val_col, marker="s")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(base_dir / "training_dynamics.png", dpi=220)
    plt.close(fig)


@torch.no_grad()
def save_post_training_multimodal_analysis(
    model,
    dataloader,
    device,
    class_names,
    output_dir: str,
    split: str,
    max_batches: Optional[int] = None,
    retrieval_top_k: int = 4,
):
    base_dir = Path(output_dir) / "multimodal_analysis" / "after_training" / split
    base_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    image_embeds = []
    cosine_rows = []
    token_rows = []
    prototype_relation = None

    for batch_idx, (inputs, targets, paths) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, aux = model(inputs, return_aux=True)

        visual_probs = torch.softmax(aux["visual_logits"], dim=1).cpu().numpy()
        text_probs = torch.softmax(aux["description_logits"], dim=1).cpu().numpy()
        fused_probs = torch.softmax(logits, dim=1).cpu().numpy()
        cosine_matrix = torch.matmul(aux["image_embedding"], aux["description_embedding_bank"].T).cpu().numpy()
        base_visual_probs = (
            torch.softmax(aux["base_visual_logits"], dim=1).cpu().numpy()
            if "base_visual_logits" in aux
            else None
        )
        enhanced_visual_probs = (
            torch.softmax(aux["enhanced_visual_logits"], dim=1).cpu().numpy()
            if "enhanced_visual_logits" in aux
            else None
        )
        fusion_gate = aux["fusion_gate"].detach().cpu().numpy() if "fusion_gate" in aux else None
        if "scale_token_weights" in aux:
            token_rows.append(aux["scale_token_weights"].detach().cpu().numpy())
        if "prototype_relation_matrix" in aux:
            prototype_relation = aux["prototype_relation_matrix"].detach().cpu().numpy()

        image_embeds.append(aux["image_embedding"].cpu().numpy())
        cosine_rows.append(cosine_matrix)

        for idx, path in enumerate(paths):
            true_label = int(targets[idx].item())
            rows.append(
                {
                    "path": path,
                    "true_label": true_label,
                    "true_class": class_names[true_label],
                    "fused_pred": class_names[int(np.argmax(fused_probs[idx]))],
                    "visual_pred": class_names[int(np.argmax(visual_probs[idx]))],
                    "text_pred": class_names[int(np.argmax(text_probs[idx]))],
                    "base_visual_pred": class_names[int(np.argmax(base_visual_probs[idx]))] if base_visual_probs is not None else None,
                    "enhanced_visual_pred": class_names[int(np.argmax(enhanced_visual_probs[idx]))] if enhanced_visual_probs is not None else None,
                    "fusion_gate": float(fusion_gate[idx, 0]) if fusion_gate is not None else None,
                    **{f"visual_{class_name}": float(visual_probs[idx, j]) for j, class_name in enumerate(class_names)},
                    **{f"text_{class_name}": float(text_probs[idx, j]) for j, class_name in enumerate(class_names)},
                    **{f"fused_{class_name}": float(fused_probs[idx, j]) for j, class_name in enumerate(class_names)},
                    **{f"cosine_{class_name}": float(cosine_matrix[idx, j]) for j, class_name in enumerate(class_names)},
                    **(
                        {f"base_visual_{class_name}": float(base_visual_probs[idx, j]) for j, class_name in enumerate(class_names)}
                        if base_visual_probs is not None
                        else {}
                    ),
                    **(
                        {f"enhanced_visual_{class_name}": float(enhanced_visual_probs[idx, j]) for j, class_name in enumerate(class_names)}
                        if enhanced_visual_probs is not None
                        else {}
                    ),
                }
            )

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(base_dir / f"{split}_multimodal_predictions.csv", index=False)

    def _mean_matrix(prefix: str):
        cols = [f"{prefix}_{class_name}" for class_name in class_names]
        return df.groupby("true_class")[cols].mean().reindex(class_names).to_numpy()

    _save_heatmap(_mean_matrix("visual"), class_names, class_names, "Mean Visual Probabilities by True Class", base_dir / "mean_visual_probabilities.png")
    _save_heatmap(_mean_matrix("text"), class_names, class_names, "Mean TF-IDF Text Probabilities by True Class", base_dir / "mean_text_probabilities.png")
    _save_heatmap(_mean_matrix("fused"), class_names, class_names, "Mean Fused Probabilities by True Class", base_dir / "mean_fused_probabilities.png")
    _save_heatmap(_mean_matrix("cosine"), class_names, class_names, "Mean Image/Text Cosine Similarity by True Class", base_dir / "mean_cosine_similarity.png")
    if any(col.startswith("base_visual_") for col in df.columns):
        _save_heatmap(
            _mean_matrix("base_visual"),
            class_names,
            class_names,
            "Mean Base Visual Probabilities by True Class",
            base_dir / "mean_base_visual_probabilities.png",
        )
    if any(col.startswith("enhanced_visual_") for col in df.columns):
        _save_heatmap(
            _mean_matrix("enhanced_visual"),
            class_names,
            class_names,
            "Mean Enhanced Visual Probabilities by True Class",
            base_dir / "mean_enhanced_visual_probabilities.png",
        )

    agreement_counts = {
        "all_same": int(((df["fused_pred"] == df["visual_pred"]) & (df["visual_pred"] == df["text_pred"])).sum()),
        "fused_matches_visual": int(((df["fused_pred"] == df["visual_pred"]) & (df["fused_pred"] != df["text_pred"])).sum()),
        "fused_matches_text": int(((df["fused_pred"] == df["text_pred"]) & (df["fused_pred"] != df["visual_pred"])).sum()),
        "fused_differs_both": int(((df["fused_pred"] != df["visual_pred"]) & (df["fused_pred"] != df["text_pred"])).sum()),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(agreement_counts.keys()), list(agreement_counts.values()), color=plt.cm.Set2(np.linspace(0.2, 0.8, len(agreement_counts))))
    ax.set_title("Branch Agreement Summary")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(base_dir / "branch_agreement_summary.png", dpi=220)
    plt.close(fig)

    if "fusion_gate" in df.columns and df["fusion_gate"].notna().any():
        gate_by_class = df.groupby("true_class")["fusion_gate"].mean().reindex(class_names)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(gate_by_class.index, gate_by_class.values, color=plt.cm.coolwarm(np.linspace(0.15, 0.85, len(class_names))))
        ax.set_title("Mean Dynamic Fusion Gate by True Class")
        ax.set_ylabel("Gate Value")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(base_dir / "fusion_gate_by_class.png", dpi=220)
        plt.close(fig)

    image_embeds_np = np.concatenate(image_embeds, axis=0)
    proto = aux["description_embedding_bank"].cpu().numpy()
    if image_embeds_np.shape[1] >= 2:
        joint = np.concatenate([image_embeds_np, proto], axis=0)
        coords = PCA(n_components=2).fit_transform(joint)
        img_coords = coords[: image_embeds_np.shape[0]]
        proto_coords = coords[image_embeds_np.shape[0] :]
        fig, ax = plt.subplots(figsize=(9, 7))
        true_indices = df["true_label"].to_numpy()
        scatter = ax.scatter(img_coords[:, 0], img_coords[:, 1], c=true_indices, cmap="tab10", alpha=0.65, s=24)
        ax.scatter(proto_coords[:, 0], proto_coords[:, 1], c=np.arange(len(class_names)), cmap="tab10", marker="X", s=220, edgecolors="black")
        for idx, class_name in enumerate(class_names):
            ax.text(proto_coords[idx, 0], proto_coords[idx, 1], class_name, fontsize=11, ha="left", va="bottom")
        ax.set_title("Image Embeddings and TF-IDF Class Prototypes (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(base_dir / "image_text_embedding_pca.png", dpi=220)
        plt.close(fig)

    if token_rows:
        token_labels = [
            "Avg f3", "Max f3", "Avg f4", "Max f4", "Avg f5", "Max f5",
            "Avg f6", "Max f6", "GeM f3", "GeM f4", "GeM f5", "GeM f6",
        ]
        token_matrix = np.concatenate(token_rows, axis=0)
        token_df = pd.DataFrame(token_matrix, columns=token_labels)
        token_df["true_class"] = df["true_class"].to_numpy()
        mean_tokens = token_df.groupby("true_class")[token_labels].mean().reindex(class_names).to_numpy()
        _save_heatmap(
            mean_tokens,
            class_names,
            token_labels,
            "Mean Scale Token Weights by True Class",
            base_dir / "mean_scale_token_weights.png",
            cmap="cividis",
        )

    if prototype_relation is not None:
        _save_heatmap(
            prototype_relation,
            class_names,
            class_names,
            "Prototype Relation Matrix",
            base_dir / "prototype_relation_matrix.png",
            cmap="magma",
        )

    cosine_cols = [f"cosine_{class_name}" for class_name in class_names]
    for class_name in class_names:
        top_rows = df.nlargest(retrieval_top_k, f"cosine_{class_name}")
        fig, axes = plt.subplots(1, len(top_rows), figsize=(4.2 * len(top_rows), 4.5))
        axes = np.atleast_1d(axes)
        for ax, (_, row) in zip(axes, top_rows.iterrows()):
            image = Image.open(row["path"]).convert("RGB")
            ax.imshow(image)
            ax.set_title(f"{row['true_class']}\ncos={row[f'cosine_{class_name}']:.2f}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"Top Images Matched to TF-IDF Prototype: {class_name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(base_dir / f"retrieval_{class_name}.png", dpi=220)
        plt.close(fig)
