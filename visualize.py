import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader

from models.agsenet_classifier import AGSENetClassifier
from data.dataset import RoadPondingDataset
from data.transforms import get_transforms, get_inverse_transform

# pytorch_grad_cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def parse_args():
    parser = argparse.ArgumentParser(description="AGSENet Classification visualizer for paper figures")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'], help='Dataset split to visualize')
    parser.add_argument('--export-image-panels', action='store_true', help='Export validation image figures with titles')
    parser.add_argument('--export-encoder-gradcam', action='store_true', help='Export encoder stage Grad-CAM figures')
    parser.add_argument('--export-decoder-gradcam', action='store_true', help='Export decoder/fusion stage Grad-CAM figures')
    parser.add_argument('--export-roc', action='store_true', help='Export multi-class ROC curve')
    parser.add_argument('--export-recall-confidence', action='store_true', help='Export recall-confidence curve')
    parser.add_argument('--gradcam-target', type=str, default='predicted', choices=['predicted', 'true'], help='Target class for Grad-CAM')
    parser.add_argument('--max-images', type=int, default=None, help='Limit the number of images to process')
    parser.add_argument('--best-checkpoint', type=str, default='auto', help='Path to weights, or auto to find best_model.pth')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_output_dirs(base_dir, split):
    out_dir = Path(base_dir) / 'visualizations' / split
    dirs = {
        'base': out_dir,
        'image_titles': out_dir / 'image_titles',
        'roc': out_dir / 'roc',
        'recall_conf': out_dir / 'recall_confidence',
        'enc_base': out_dir / 'encoder_gradcam',
        'dec_base': out_dir / 'decoder_gradcam'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    for d in [dirs['enc_base'], dirs['dec_base']]:
        (d / 'combined').mkdir(exist_ok=True)
        
    for i in range(1, 7):
        (dirs['enc_base'] / f'stage{i}').mkdir(exist_ok=True)
        
    for i in range(1, 5):
        (dirs['dec_base'] / f'stage{i}').mkdir(exist_ok=True)
        
    return dirs

def generate_figure_panel(original_img, heatmaps, titles, combined_save_path):
    n_panels = 1 + len(heatmaps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')
    
    for i, (hm, title) in enumerate(zip(heatmaps, titles)):
        axes[i+1].imshow(hm)
        axes[i+1].set_title(title, fontsize=14)
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig(combined_save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_roc_curve(y_true, y_probs, class_names, save_dir):
    n_classes = len(class_names)
    y_true_bin = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_true_bin[i, label] = 1.0
        
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)
             
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linestyle=':', linewidth=4)
             
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:0.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Multi-class ROC implementation', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png', dpi=300)
    plt.close()

def plot_recall_confidence(y_true, y_probs, y_preds, class_names, save_dir):
    thresholds = np.linspace(0.0, 1.0, 100)
    recalls = []
    
    for t in thresholds:
        accepted = y_probs.max(axis=1) >= t
        if not np.any(accepted):
            recalls.append(0.0)
            continue
            
        correct = (y_preds[accepted] == y_true[accepted])
        recall_at_t = np.sum(correct) / len(y_true) 
        recalls.append(recall_at_t)
        
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, lw=2, color='blue', label='Overall Recall vs Confidence')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Confidence Threshold', fontsize=14)
    plt.ylabel('Recall (True Positives / All Positives)', fontsize=14)
    plt.title('Recall vs Confidence Threshold', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'recall_confidence_curve.png', dpi=300)
    plt.close()
    
    # Save CSV
    df = pd.DataFrame({'Threshold': thresholds, 'Recall': recalls})
    df.to_csv(save_dir / 'recall_confidence_data.csv', index=False)

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dirs = create_output_dirs(config['output_dir'], args.split)
    
    # Checkpoint
    if args.best_checkpoint == 'auto':
        ckpt_path = Path(config['output_dir']) / 'checkpoints' / 'best_model.pth'
    else:
        ckpt_path = Path(args.best_checkpoint)
        
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Dataset
    val_ratio = config.get('val_split_ratio', 0.2)
    test_ratio = config.get('test_split_ratio', 0.1)
    transform = get_transforms(config['image_size'], split=args.split)
    
    dataset = RoadPondingDataset(
        config['data_path'], transform=transform, split=args.split, 
        val_ratio=val_ratio, test_ratio=test_ratio
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.get('num_workers', 4))
    inv_transform = get_inverse_transform()
    class_names = dataset.classes
    
    # Model
    model = AGSENetClassifier(
        in_ch=3, out_ch=config['num_classes'], 
        base_ch=config['model_channels'], dropout=config['dropout']
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # CAM definitions
    encoder_layers = [model.en_1, model.en_2, model.en_3, model.en_4, model.en_5, model.en_6]
    decoder_layers = [model.proj_6, model.ssie_5, model.ssie_4, model.ssie_3]
    
    # Output structure
    metadata = []
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    
    num_processed = 0
    tqdm_loader = tqdm(dataloader, desc=f"Processing {args.split}")
    
    for i, (inputs, targets) in enumerate(tqdm_loader):
        if args.max_images and num_processed >= args.max_images:
            break
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        true_label = targets.item()
        true_name = class_names[true_label]
        
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probs)
            pred_name = class_names[pred_label]
            conf = probs[pred_label]
            
        y_true_list.append(true_label)
        y_pred_list.append(pred_label)
        y_prob_list.append(probs)
        
        is_correct = (true_label == pred_label)
        
        # Denorm Original Image
        img_denorm = inv_transform(inputs[0])
        rgb_img = np.float32(img_denorm) / 255.0
        orig_uint8 = np.uint8(255 * rgb_img)
        
        base_name = f"idx{i:05d}_T-{true_name}_P-{pred_name}"
        
        # 1. Titled Image Panels
        if args.export_image_panels:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rgb_img)
            color = 'green' if is_correct else 'red'
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
            ax.set_title(f"GT: {true_name} | Pred: {pred_name} | Conf: {conf:.2f}",
                         fontsize=14, color=color, pad=15)
            ax.axis('off')
            title_path = dirs['image_titles'] / f"{base_name}_titled.png"
            plt.savefig(title_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
        # Target assignment for CAM
        cam_target = [ClassifierOutputTarget(targets.item() if args.gradcam_target == 'true' else pred_label)]
        
        # 2. Encoder Grad-CAM
        enc_overlay_files = []
        if args.export_encoder_gradcam:
            enc_heatmaps = []
            for j, layer in enumerate(encoder_layers):
                with GradCAM(model=model, target_layers=[layer]) as cam:
                    gray_cam = cam(input_tensor=inputs, targets=cam_target)[0]
                overlay = show_cam_on_image(rgb_img, gray_cam, use_rgb=True)
                
                # Save stage-specific single overlay
                stage_path = dirs['enc_base'] / f'stage{j+1}' / f"{base_name}_enc{j+1}.png"
                cv2.imwrite(str(stage_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                enc_overlay_files.append(str(stage_path))
                
                enc_heatmaps.append(overlay)
                
            combined_path = dirs['enc_base'] / 'combined' / f"{base_name}_encoder_panel.png"
            generate_figure_panel(rgb_img, enc_heatmaps, [f"Encoder {j+1}" for j in range(6)], combined_path)
            
        # 3. Decoder Grad-CAM
        dec_overlay_files = []
        if args.export_decoder_gradcam:
            dec_heatmaps = []
            for j, layer in enumerate(decoder_layers):
                with GradCAM(model=model, target_layers=[layer]) as cam:
                    gray_cam = cam(input_tensor=inputs, targets=cam_target)[0]
                overlay = show_cam_on_image(rgb_img, gray_cam, use_rgb=True)
                
                stage_path = dirs['dec_base'] / f'stage{j+1}' / f"{base_name}_dec{j+1}.png"
                cv2.imwrite(str(stage_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                dec_overlay_files.append(str(stage_path))
                
                dec_heatmaps.append(overlay)
                
            combined_path = dirs['dec_base'] / 'combined' / f"{base_name}_decoder_panel.png"
            # Decoder names reflect fusion steps: 1=Proj6, 2=SSIE5, 3=SSIE4, 4=SSIE3
            generate_figure_panel(rgb_img, dec_heatmaps, [f"Decoder {j+1}" for j in range(4)], combined_path)
            
        metadata.append({
            'index': i,
            'split': args.split,
            'true_label': true_name,
            'pred_label': pred_name,
            'confidence': f"{conf:.4f}",
            'correct': is_correct,
            'checkpoint_used': str(ckpt_path),
            'titled_image': str(dirs['image_titles'] / f"{base_name}_titled.png") if args.export_image_panels else None,
            'enc_panel': str(dirs['enc_base'] / 'combined' / f"{base_name}_encoder_panel.png") if args.export_encoder_gradcam else None,
            'dec_panel': str(dirs['dec_base'] / 'combined' / f"{base_name}_decoder_panel.png") if args.export_decoder_gradcam else None,
        })
        num_processed += 1

    y_true_np = np.array(y_true_list)
    y_probs_np = np.array(y_prob_list)
    y_preds_np = np.array(y_pred_list)
    
    # 4. ROC Curve
    if args.export_roc:
        print("Generating ROC Curves...")
        plot_roc_curve(y_true_np, y_probs_np, class_names, dirs['roc'])
        
    # 5. Recall-Confidence
    if args.export_recall_confidence:
        print("Generating Recall-Confidence Curves...")
        plot_recall_confidence(y_true_np, y_probs_np, y_preds_np, class_names, dirs['recall_conf'])
        
    # Meta CSV
    print("Saving Metadata CSV...")
    df_meta = pd.DataFrame(metadata)
    df_meta.to_csv(dirs['base'] / f'metadata_{args.split}.csv', index=False)
    
    print(f"\nCompleted! Visualizations saved to {dirs['base']}")

if __name__ == '__main__':
    main()
