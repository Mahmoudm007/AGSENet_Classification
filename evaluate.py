import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.agsenet_classifier import AGSENetClassifier
from data import RoadPondingDataset, get_transforms
from utils import MetricTracker, set_seed
from utils.visualization import GradCAMExporter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AGSENet Classification Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model weights')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='val', help='Which split to evaluate')
    parser.add_argument('--export-gradcam', action='store_true', help='Export Grad-CAM for the entire split')
    parser.add_argument('--gradcam-target', type=str, choices=['predicted', 'true'], default='predicted', help='Target class for Grad-CAM')
    return parser.parse_args()

@torch.no_grad()
def evaluate(model, dataloader, device, class_names):
    model.eval()
    metric_tracker = MetricTracker(class_names=class_names)
    
    features = []
    def hook(module, input, output):
        features.append(input[0].detach().cpu().numpy())
    handle = model.classifier[0].register_forward_hook(hook)
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for inputs, targets, paths in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        metric_tracker.update(preds, targets, probs)
        
    handle.remove()
    all_features = np.concatenate(features, axis=0)
    return metric_tracker, all_features

def plot_tsne(features, targets, preds, class_names, out_dir):
    print("Computing t-SNE for clustering plot...")
    perplexity = min(30, max(5, len(features) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(features)
    
    # 1. Plot all classes together
    plt.figure(figsize=(12, 10))
    correct_idx = (targets == preds)
    incorrect_idx = (targets != preds)
    
    palette = sns.color_palette("husl", len(class_names))
    
    for i, class_name in enumerate(class_names):
        c_idx = correct_idx & (targets == i)
        plt.scatter(embedded[c_idx, 0], embedded[c_idx, 1], label=f'{class_name} (Correct)',
                    color=palette[i], alpha=0.7, marker='o', s=50)
        
        i_idx = incorrect_idx & (targets == i)
        if np.any(i_idx):
            plt.scatter(embedded[i_idx, 0], embedded[i_idx, 1], label=f'{class_name} (Incorrect)',
                        color=palette[i], alpha=1.0, marker='X', s=100, edgecolor='black')
            
    plt.title('t-SNE Clustering (All Classes)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_clusters_tsne_all.png'), dpi=300)
    plt.close()
    
    # 2. Plot between 2 classes per image
    import itertools
    for combo in itertools.combinations(range(len(class_names)), 2):
        c1, c2 = combo
        plt.figure(figsize=(10, 8))
        
        for i in [c1, c2]:
            c_idx = correct_idx & (targets == i)
            if np.any(c_idx):
                plt.scatter(embedded[c_idx, 0], embedded[c_idx, 1], label=f'{class_names[i]} (Correct)',
                            color=palette[i], alpha=0.7, marker='o', s=60)
            
            i_idx = incorrect_idx & (targets == i)
            if np.any(i_idx):
                plt.scatter(embedded[i_idx, 0], embedded[i_idx, 1], label=f'{class_names[i]} (Incorrect)',
                            color=palette[i], alpha=1.0, marker='X', s=120, edgecolor='black')
                
        plt.title(f't-SNE: {class_names[c1]} vs {class_names[c2]}', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'feature_clusters_tsne_{c1}_vs_{c2}.png'), dpi=300)
        plt.close()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset
    data_path = config['data_path']
    img_size = config['image_size']
    merge_classes = config.get('merge_classes', True)
    
    transform = get_transforms(img_size, split=args.split)
    dataset = RoadPondingDataset(data_path, transform=transform, split=args.split, merge_classes=merge_classes)
    
    dataloader = DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=False, 
        num_workers=config.get('num_workers', 4), pin_memory=True
    )
    
    # Model Setup
    model = AGSENetClassifier(
        in_ch=3, 
        out_ch=config['num_classes'], 
        base_ch=config['model_channels'], 
        dropout=config['dropout']
    ).to(device)
    
    # Load Weights
    print(f"Loading weights from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # Evaluate
    tracker, features = evaluate(model, dataloader, device, dataset.classes)
    
    metrics = tracker.compute()
    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    print("\nClassification Report:")
    print(tracker.classification_report())
    
    # Save Confusion Matrix
    cm = tracker.confusion_matrix()
    cm_df = pd.DataFrame(cm, index=dataset.classes, columns=dataset.classes)
    out_dir = config['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    cm_csv_path = os.path.join(out_dir, f"confusion_matrix_{args.split}.csv")
    cm_df.to_csv(cm_csv_path)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({args.split})')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{args.split}.png"), dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to {cm_csv_path}")
    
    # TSNE Plot
    targets_np = np.array(tracker.all_targets)
    preds_np = np.array(tracker.all_preds)
    plot_tsne(features, targets_np, preds_np, dataset.classes, out_dir)
    
    # Save base predictions
    results_df = pd.DataFrame({
        'True_Label': tracker.all_targets,
        'Predicted_Label': tracker.all_preds
    })
    results_df.to_csv(os.path.join(out_dir, f"predictions_{args.split}.csv"), index=False)
    
    # Grad-CAM Export
    if args.export_gradcam:
        print("\nStarting full-dataset Grad-CAM export...")
        target_layer = config.get('gradcam_target_layer', 'auto')
        exporter = GradCAMExporter(model, out_dir, device, target_layer=target_layer)
        
        # DataLoader needs batch size 1 or handled batch size for grad-cam
        cam_loader = DataLoader(
            dataset, batch_size=config['batch_size'], shuffle=False, 
            num_workers=config.get('num_workers', 4), pin_memory=True
        )
        use_true = (args.gradcam_target == 'true')
        exporter.export_dataset(cam_loader, dataset.classes, args.split, use_true_label=use_true)

if __name__ == "__main__":
    main()
