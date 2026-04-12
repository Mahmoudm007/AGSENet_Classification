import os
import yaml
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.agsenet_classifier import AGSENetClassifier
from data.dataset import RoadPondingDataset
from data.transforms import get_transforms
from utils import get_loss_function, MetricTracker, CSVLogger, AverageMeter, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train AGSENet Classification Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, metric_tracker):
    model.train()
    losses = AverageMeter()
    metric_tracker.reset()
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets, _ in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        
        # Optional gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        losses.update(loss.item(), inputs.size(0))
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        metric_tracker.update(preds, targets, probs)
        pbar.set_postfix({'loss': f"{losses.avg:.4f}"})
        
    metrics = metric_tracker.compute()
    metrics['loss'] = losses.avg
    return metrics

@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device, metric_tracker, epoch, out_dir):
    model.eval()
    losses = AverageMeter()
    metric_tracker.reset()
    
    sample_losses = []
    
    pbar = tqdm(dataloader, desc="Validation")
    for inputs, targets, paths in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute per-sample unreduced loss for highest loss tracking
            per_sample_loss = F.cross_entropy(outputs, targets, reduction='none')
            
        losses.update(loss.item(), inputs.size(0))
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        metric_tracker.update(preds, targets, probs)
        pbar.set_postfix({'loss': f"{losses.avg:.4f}"})
        
        # Store metadata for tracking highest losses
        for b_idx in range(inputs.size(0)):
            sample_losses.append({
                'path': paths[b_idx],
                'loss': per_sample_loss[b_idx].item(),
                'pred': preds[b_idx].item(),
                'true': targets[b_idx].item(),
                'prob': probs[b_idx, preds[b_idx]].item()
            })
            
    # Save highest loss examples
    sample_losses.sort(key=lambda x: x['loss'], reverse=True)
    top_losses = sample_losses[:20] # track top 20 worst predictions
    
    hl_dir = os.path.join(out_dir, "high_loss_samples", f"epoch_{epoch}")
    os.makedirs(hl_dir, exist_ok=True)
    
    for rank, sl in enumerate(top_losses):
        ext = os.path.splitext(sl['path'])[1]
        new_name = f"Rank{rank+1}_Loss{sl['loss']:.3f}_P{sl['pred']}_T{sl['true']}{ext}"
        try:
            shutil.copy(sl['path'], os.path.join(hl_dir, new_name))
        except:
            pass
            
    # Also save CSV of highest losses
    pd.DataFrame(top_losses).to_csv(os.path.join(hl_dir, "top_losses.csv"), index=False)
        
    metrics = metric_tracker.compute()
    metrics['loss'] = losses.avg
    return metrics

def main():
    args = parse_args()
    config = load_config(args.config)
    
    set_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset Preparation
    data_path = config['data_path']
    img_size = config['image_size']
    merge_classes = config.get('merge_classes', True)
    
    train_transform = get_transforms(img_size, split='train')
    val_transform = get_transforms(img_size, split='val')
    
    train_dataset = RoadPondingDataset(data_path, transform=train_transform, split='train', merge_classes=merge_classes)
    val_dataset = RoadPondingDataset(data_path, transform=val_transform, split='val', merge_classes=merge_classes)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, 
        num_workers=config.get('num_workers', 4), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, 
        num_workers=config.get('num_workers', 4), pin_memory=True
    )
    
    class_frequencies = train_dataset.get_class_frequencies()
    criterion = get_loss_function(config, class_frequencies=class_frequencies).to(device)
    
    # Model Setup
    model = AGSENetClassifier(
        in_ch=3, 
        out_ch=config['num_classes'], 
        base_ch=config['model_channels'], 
        dropout=config['dropout']
    ).to(device)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    pd.DataFrame({'total_learnable_parameters': [total_params]}).to_csv(os.path.join(config['output_dir'], 'model_parameters.csv'), index=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    
    if config.get('scheduler_type') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
    scaler = torch.amp.GradScaler('cuda')
    
    metric_tracker = MetricTracker(class_names=train_dataset.classes)
    logger = CSVLogger(config['output_dir'])
    
    best_val_loss = float('inf')
    os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n[{epoch}/{config['epochs']}] Training...")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, metric_tracker)
        logger.print_metrics(epoch, train_metrics, is_val=False)
        
        if epoch % config.get('validate_every_n_epochs', 1) == 0:
            val_metrics = validate_epoch(model, val_loader, criterion, device, metric_tracker, epoch, config['output_dir'])
            logger.print_metrics(epoch, val_metrics, is_val=True)
            
            # Combine dicts for logging
            combined_metrics = {}
            for k, v in train_metrics.items():
                combined_metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                combined_metrics[f"val_{k}"] = v
            combined_metrics['lr'] = optimizer.param_groups[0]['lr']
            
            logger.log(epoch, combined_metrics)
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_path = os.path.join(config['output_dir'], 'checkpoints', 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model with Validation Loss: {best_val_loss:.4f}")
                
        scheduler.step()
        
    # Save final model
    final_path = os.path.join(config['output_dir'], 'checkpoints', 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print("Training Completed.")
    
    print("\nRunning comprehensive evaluation and explainability on best model...")
    import subprocess
    best_path = os.path.join(config['output_dir'], 'checkpoints', 'best_model.pth')
    if os.path.exists(best_path):
        try:
            print("--> Running evaluate.py")
            subprocess.run(["python", "evaluate.py", "--config", args.config, "--checkpoint", best_path, "--split", "val", "--export-gradcam"], check=True)
            print("--> Running visualize.py")
            subprocess.run(["python", "visualize.py", "--config", args.config, "--split", "val", "--export-image-panels", "--export-encoder-gradcam", "--export-decoder-gradcam", "--export-roc", "--export-recall-confidence", "--best-checkpoint", best_path], check=True)
        except Exception as e:
            print(f"Error during automatic evaluation: {e}")

if __name__ == "__main__":
    main()
