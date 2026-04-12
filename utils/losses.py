import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_class_weights(class_frequencies, mode='inverse', beta=0.9999):
    """
    Computes class weights from class training frequencies.
    """
    freq = np.array(class_frequencies, dtype=np.float32)
    # Avoid division by zero
    freq = np.maximum(freq, 1.0)
    
    if mode == 'inverse':
        weights = 1.0 / freq
    elif mode == 'balanced':
        total_samples = np.sum(freq)
        num_classes = len(freq)
        weights = total_samples / (num_classes * freq)
    elif mode == 'median_frequency':
        median = np.median(freq)
        weights = median / freq
    elif mode == 'normalized_inverse':
        weights = 1.0 / freq
        weights = weights / np.sum(weights) * len(freq)
    elif mode == 'effective_num':
        effective_num = 1.0 - np.power(beta, freq)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
        weights = weights / np.sum(weights) * len(freq)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
        
    return torch.tensor(weights, dtype=torch.float32)

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # can be a tensor of class weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = focal_loss * at
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(config, class_frequencies=None):
    loss_name = config.get("loss_name", "cross_entropy")
    label_smoothing = config.get("label_smoothing", 0.0)
    
    weights = None
    if config.get("use_class_weights", False) and class_frequencies is not None:
        mode = config.get("class_weight_mode", "inverse")
        if mode != "manual":
            weights = compute_class_weights(
                class_frequencies,
                mode=mode,
                beta=float(config.get("class_weight_beta", 0.9999)),
            )
            print(f"Computed Class Weights ({mode}): {weights.tolist()}")
    
    if loss_name in ["cross_entropy", "weighted_cross_entropy"]:
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        
    elif loss_name in ["focal", "class_balanced_focal"]:
        gamma = config.get("focal_gamma", 2.0)
        # alpha is only used in focal loss if use_class_weights is enabled
        alpha_weights = weights if config.get("use_class_weights", False) else None
        return FocalLoss(alpha=alpha_weights, gamma=gamma)
        
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")
