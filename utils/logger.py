import os
import csv
import json
from datetime import datetime

class CSVLogger:
    def __init__(self, output_dir, filename="training_log.csv"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, filename)

    def log(self, epoch, metrics_dict):
        row = {'epoch': epoch, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        row.update(metrics_dict)

        file_exists = os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        with open(os.path.join(self.output_dir, "metrics.jsonl"), 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + "\n")
            
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
