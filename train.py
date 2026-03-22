import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.agsenet_classifier import AGSENetClassifier
from data.dataset import RoadPondingDataset
from data.transforms import get_transforms
from utils import get_loss_function, MetricTracker, CSVLogger, AverageMeter, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train AGSENet Classification Model")
    parser.add_config = parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, metric_tracker):
    model.train()
    losses = AverageMeter()
    metric_tracker.reset()
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
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
def validate_epoch(model, dataloader, criterion, device, metric_tracker):
    model.eval()
    losses = AverageMeter()
    metric_tracker.reset()
    
    pbar = tqdm(dataloader, desc="Validation")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        losses.update(loss.item(), inputs.size(0))
        
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        metric_tracker.update(preds, targets, probs)
        pbar.set_postfix({'loss': f"{losses.avg:.4f}"})
        
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
    val_ratio = config.get('val_split_ratio', 0.2)
    test_ratio = config.get('test_split_ratio', 0.1)
    
    train_transform = get_transforms(img_size, split='train')
    val_transform = get_transforms(img_size, split='val')
    
    train_dataset = RoadPondingDataset(data_path, transform=train_transform, split='train', val_ratio=val_ratio, test_ratio=test_ratio)
    val_dataset = RoadPondingDataset(data_path, transform=val_transform, split='val', val_ratio=val_ratio, test_ratio=test_ratio)
    
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    
    if config.get('scheduler_type') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
    scaler = torch.amp.GradScaler('cuda')
    
    metric_tracker = MetricTracker(class_names=train_dataset.classes)
    logger = CSVLogger(config['output_dir'])
    
    best_val_f1 = 0.0
    os.makedirs(os.path.join(config['output_dir'], 'checkpoints'), exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n[{epoch}/{config['epochs']}] Training...")
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device, metric_tracker)
        logger.print_metrics(epoch, train_metrics, is_val=False)
        
        if epoch % config.get('validate_every_n_epochs', 1) == 0:
            val_metrics = validate_epoch(model, val_loader, criterion, device, metric_tracker)
            logger.print_metrics(epoch, val_metrics, is_val=True)
            
            # Combine dicts for logging
            combined_metrics = {}
            for k, v in train_metrics.items():
                combined_metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                combined_metrics[f"val_{k}"] = v
            combined_metrics['lr'] = optimizer.param_groups[0]['lr']
            
            logger.log(epoch, combined_metrics)
            
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_path = os.path.join(config['output_dir'], 'checkpoints', 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model with Validation Macro F1: {best_val_f1:.4f}")
                
        scheduler.step()
        
    # Save final model
    final_path = os.path.join(config['output_dir'], 'checkpoints', 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print("Training Completed.")

if __name__ == "__main__":
    main()
