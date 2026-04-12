import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from data import get_transforms
from utils import build_model, load_checkpoint_payload, load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Inference AGSENet Classification Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights")
    parser.add_argument("--input", type=str, required=True, help="Path to image file or directory")
    parser.add_argument("--output_csv", type=str, default="inference_results.csv", help="Path to save output CSV")
    return parser.parse_args()


@torch.no_grad()
def infer_image(model, image_path, transform, device, class_names):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        outputs = model(tensor)

    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    row = {
        "image_file": str(image_path),
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
    }
    for i, class_name in enumerate(class_names):
        row[f"prob_{class_name}"] = float(probs[i])
    return row


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path, device)
    runtime_config = checkpoint_payload.get("config", config)
    class_names = checkpoint_payload.get("class_names")
    if class_names is None:
        raise ValueError("Checkpoint does not contain class_names. Re-train or use a newer checkpoint.")

    model, _ = build_model(config, class_names, device, checkpoint_payload=checkpoint_payload)
    missing, unexpected = load_model_weights(model, checkpoint_payload, strict=True)
    if missing or unexpected:
        print(f"Checkpoint load warnings | missing={missing} | unexpected={unexpected}")
    model.eval()

    transform = get_transforms(runtime_config["image_size"], split="test")

    input_path = Path(args.input)
    if input_path.is_file():
        result = infer_image(model, input_path, transform, device, class_names)
        print(f"\n--- Inference: {input_path.name} ---")
        print(f"Predicted Class: {result['predicted_class']} ({result['confidence']:.4f})")
        for class_name in class_names:
            print(f"  {class_name}: {result[f'prob_{class_name}']:.4f}")
        return

    if input_path.is_dir():
        image_paths = [
            path
            for path in input_path.rglob("*.*")
            if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]
        rows = []
        for image_path in tqdm(image_paths, desc="Inferencing"):
            try:
                rows.append(infer_image(model, image_path, transform, device, class_names))
            except Exception as exc:
                print(f"Skipping {image_path}: {exc}")

        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"Saved {len(rows)} inference results to {args.output_csv}")
        return

    raise FileNotFoundError(f"Invalid input path: {args.input}")


if __name__ == "__main__":
    main()
