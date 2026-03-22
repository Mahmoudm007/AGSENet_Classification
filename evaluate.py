import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
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
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        metric_tracker.update(preds, targets, probs)
        
    return metric_tracker

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
    val_ratio = config.get('val_split_ratio', 0.2)
    test_ratio = config.get('test_split_ratio', 0.1)
    
    transform = get_transforms(img_size, split=args.split)
    dataset = RoadPondingDataset(data_path, transform=transform, split=args.split, val_ratio=val_ratio, test_ratio=test_ratio)
    
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
    tracker = evaluate(model, dataloader, device, dataset.classes)
    
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
    print(f"Confusion matrix saved to {cm_csv_path}")
    
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
