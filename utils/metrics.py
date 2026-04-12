import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix

class MetricTracker:
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.reset()
        
    def reset(self):
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []
        
    def update(self, preds, targets, probs=None):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        self.all_preds.extend(preds)
        self.all_targets.extend(targets)
        
        if probs is not None:
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            self.all_probs.extend(probs)
            
    def compute(self):
        if not self.all_targets:
            return {}
            
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)
        
        # Core Metrics
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        weighted_f1 = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )[2]
        
        metrics = {
            'acc': acc,
            'bal_acc': bal_acc,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'weighted_f1': weighted_f1
        }

        if self.all_probs and self.class_names:
            y_prob = np.array(self.all_probs)
            num_classes = y_prob.shape[1]
            for k in [2, 3]:
                if num_classes < k:
                    continue
                topk = np.argpartition(y_prob, -k, axis=1)[:, -k:]
                topk_correct = [int(label in preds) for label, preds in zip(y_true, topk)]
                metrics[f'top_{k}_acc'] = float(np.mean(topk_correct))
        
        return metrics
        
    def classification_report(self):
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)
        
        names = self.class_names if self.class_names else None
        
        # If true class set is a subset of 0..4 it might crash without setting labels
        labels = np.arange(len(self.class_names)) if self.class_names else None
        
        return classification_report(y_true, y_pred, target_names=names, labels=labels, zero_division=0)
        
    def confusion_matrix(self):
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)
        labels = np.arange(len(self.class_names)) if self.class_names else None
        return confusion_matrix(y_true, y_pred, labels=labels)
