import os
import csv
from datetime import datetime

class CSVLogger:
    def __init__(self, output_dir, filename="training_log.csv"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, filename)
        
        self.headers_written = False
        
    def log(self, epoch, metrics_dict, is_val=False):
        prefix = "val_" if is_val else "train_"
        flattened_dict = {'epoch': epoch, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        for k, v in metrics_dict.items():
            flattened_dict[prefix + k] = v
            
        file_exists = os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0
        
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Extract headers from the keys
            headers = list(flattened_dict.keys())
            
            # Since train and val are logged separately we just use a unified scheme
            # For simplicity, we assume train and val log the same keys mostly, or we rewrite headers
            # A better approach: we just log a dictionary per row and use DictWriter but keys can vary
            pass

        # Simple dictionary string append instead of rigid CSV
        with open(os.path.join(self.output_dir, "metrics.jsonl"), 'a') as f:
            import json
            flattened_dict['split'] = 'val' if is_val else 'train'
            f.write(json.dumps(flattened_dict) + "\n")
            
    def print_metrics(self, epoch, metrics, is_val=False):
        split = "VAL" if is_val else "TRAIN"
        s = f"Epoch {epoch:03d} | {split} | "
        p_list = []
        for k, v in metrics.items():
            if isinstance(v, float):
                p_list.append(f"{k}: {v:.4f}")
            else:
                p_list.append(f"{k}: {v}")
        print(s + " | ".join(p_list))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
