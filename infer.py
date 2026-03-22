import os
import argparse
import yaml
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from models.agsenet_classifier import AGSENetClassifier
from data import get_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Inference AGSENet Classification Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to image file or directory')
    parser.add_argument('--output_csv', type=str, default='inference_results.csv', help='Path to save output CSV')
    return parser.parse_args()

@torch.no_grad()
def infer_directory(model, input_dir, transform, device, class_names):
    model.eval()
    input_path = Path(input_dir)
    image_paths = list(input_path.rglob("*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
    
    results = []
    
    for img_p in tqdm(image_paths, desc="Inferencing"):
        try:
            image = Image.open(img_p).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_p}: {e}")
            continue
            
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(tensor)
            
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx]
        
        row = {
            'image_file': str(img_p),
            'predicted_class': pred_class,
            'confidence': confidence
        }
        for i, cname in enumerate(class_names):
            row[f'prob_{cname}'] = probs[i]
            
        results.append(row)
        
    return results

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # We need the class names in correct order. Ideally these are saved in config or recovered from training data.
    # We will just read the training data folders since it's the source of truth if config doesn't have it.
    data_path = Path(config['data_path'])
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    # Model Setup
    model = AGSENetClassifier(
        in_ch=3, 
        out_ch=len(class_names), 
        base_ch=config['model_channels'], 
        dropout=config['dropout']
    ).to(device)
    
    print(f"Loading weights from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    transform = get_transforms(config['image_size'], split='test')
    
    input_p = Path(args.input)
    if input_p.is_file():
        # Single image
        import numpy as np
        image = Image.open(input_p).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.amp.autocast('cuda'):
            outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        
        print(f"\n--- Inference: {input_p.name} ---")
        print(f"Predicted Class: {class_names[pred_idx]} ({probs[pred_idx]:.4f})")
        for i, cname in enumerate(class_names):
            print(f"  {cname}: {probs[i]:.4f}")
            
    elif input_p.is_dir():
        import numpy as np
        results = infer_directory(model, input_p, transform, device, class_names)
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(results)} inference results to {args.output_csv}")
    else:
        print(f"Invalid input path: {args.input}")

if __name__ == "__main__":
    main()
