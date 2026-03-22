import os
import csv
import torch
import numpy as np
import cv2
from tqdm import tqdm
from data.transforms import get_inverse_transform

# Uses the standard grad-cam library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class GradCAMExporter:
    """
    Handles extracting Grad-CAM heatmaps for an entire dataset split and saving results.
    """
    def __init__(self, model, output_dir, device='cuda', target_layer=None):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # If target layer is 'auto' or None, pick the final SSIE fusion block's projection
        if target_layer == 'auto' or target_layer is None:
            # We target the last fusion convolution in SSIE 3
            self.target_layers = [model.ssie_3.fusion_conv.conv]
        else:
            self.target_layers = [target_layer]
            
        self.cam = GradCAM(model=model, target_layers=self.target_layers)
        self.inverse_transform = get_inverse_transform()

    def export_dataset(self, dataloader, class_names, split_name, use_true_label=False):
        """
        Exports Grad-CAM over an entire split.
        If use_true_label is False, generates heatmap for the PREDICTED class.
        Otherwise, for the TRUE class.
        """
        out_root = os.path.join(self.output_dir, "gradcam", split_name)
        os.makedirs(out_root, exist_ok=True)
        
        # Setup class folders
        for cls in class_names:
            os.makedirs(os.path.join(out_root, cls), exist_ok=True)
            
        csv_path = os.path.join(out_root, f"gradcam_{split_name}_results.csv")
        csv_exists = os.path.exists(csv_path)
        
        f = open(csv_path, 'a', newline='')
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "image_index", "split", "true_label", "true_class_name", 
                "predicted_label", "predicted_class_name", "correct", 
                "overlay_path", "heatmap_path", "original_path"
            ])

        self.model.eval()
        idx_counter = 0

        print(f"Starting Grad-CAM export for {split_name} (Total Batches: {len(dataloader)})")

        for inputs, targets in tqdm(dataloader, desc=f"Grad-CAM {split_name}"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            with torch.no_grad():
                preds = self.model(inputs)
                probs = torch.softmax(preds, dim=1)
                predicted_labels = torch.argmax(probs, dim=1).cpu().numpy()
            
            # grad-cam library processes per batch
            # target the predicted class or true class
            if use_true_label:
                cam_targets = [ClassifierOutputTarget(val) for val in targets.cpu().numpy()]
            else:
                cam_targets = [ClassifierOutputTarget(val) for val in predicted_labels]

            grayscale_cams = self.cam(input_tensor=inputs, targets=cam_targets)

            # Save each image in the batch
            for b in range(inputs.size(0)):
                true_label = targets[b].item()
                pred_label = predicted_labels[b]
                is_correct = (true_label == pred_label)
                
                true_class_name = class_names[true_label]
                pred_class_name = class_names[pred_label]

                # Denormalize image for visualization [H, W, 3] Float tensor in [0, 1]
                img_denorm = self.inverse_transform(inputs[b])
                rgb_img = np.float32(img_denorm) / 255.0

                # Ensure dimension
                gray_cam = grayscale_cams[b, :]
                
                # Apply overlay
                overlay = show_cam_on_image(rgb_img, gray_cam, use_rgb=True)
                
                # Apply jet colormap separately for standalone heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * gray_cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                # Original RGB image (uint8)
                orig_uint8 = np.uint8(255 * rgb_img)
                
                base_name = f"img_{idx_counter:06d}_T{true_class_name}_P{pred_class_name}"
                cls_folder = os.path.join(out_root, true_class_name)
                
                orig_path = os.path.join(cls_folder, base_name + "_original.png")
                heatmap_path = os.path.join(cls_folder, base_name + "_heatmap.png")
                overlay_path = os.path.join(cls_folder, base_name + "_overlay.png")
                
                # Convert back to BGR for cv2 saving
                cv2.imwrite(orig_path, cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                writer.writerow([
                    idx_counter, split_name, true_label, true_class_name,
                    pred_label, pred_class_name, is_correct,
                    overlay_path, heatmap_path, orig_path
                ])
                
                idx_counter += 1

        f.close()
        print(f"Grad-CAM export complete! Saved to {out_root}")
