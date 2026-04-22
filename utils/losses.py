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


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax for long-tailed classification.
    """

    def __init__(self, class_frequencies, label_smoothing=0.0):
        super().__init__()
        freq = np.array(class_frequencies, dtype=np.float32)
        freq = np.maximum(freq, 1.0)
        priors = freq / np.sum(freq)
        self.register_buffer("log_prior", torch.log(torch.tensor(priors, dtype=torch.float32)))
        self.label_smoothing = float(label_smoothing)

    def forward(self, inputs, targets):
        adjusted_logits = inputs + self.log_prior.unsqueeze(0)
        return F.cross_entropy(adjusted_logits, targets, label_smoothing=self.label_smoothing)


class LogitAdjustedCrossEntropyLoss(nn.Module):
    """
    Logit-adjusted cross-entropy from long-tail classification literature.
    """

    def __init__(self, class_frequencies, tau=1.0, label_smoothing=0.0):
        super().__init__()
        freq = np.array(class_frequencies, dtype=np.float32)
        freq = np.maximum(freq, 1.0)
        priors = freq / np.sum(freq)
        self.register_buffer("adjustment", torch.log(torch.tensor(priors, dtype=torch.float32)))
        self.tau = float(tau)
        self.label_smoothing = float(label_smoothing)

    def forward(self, inputs, targets):
        adjusted_logits = inputs + (self.tau * self.adjustment.unsqueeze(0))
        return F.cross_entropy(adjusted_logits, targets, label_smoothing=self.label_smoothing)


def supervised_contrastive_loss(features, targets, temperature=0.07):
    if features is None or targets is None or features.ndim != 2 or features.shape[0] < 2:
        if isinstance(features, torch.Tensor):
            return features.new_tensor(0.0)
        return torch.tensor(0.0)

    features = F.normalize(features, dim=1)
    targets = targets.view(-1)
    device = features.device
    logits = torch.matmul(features, features.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(logits, device=device) - torch.eye(logits.size(0), device=device)
    positive_mask = targets.unsqueeze(0).eq(targets.unsqueeze(1)).float() * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positives_per_row = positive_mask.sum(dim=1)
    valid_rows = positives_per_row > 0
    if not torch.any(valid_rows):
        return features.new_tensor(0.0)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positives_per_row.clamp_min(1.0)
    return -mean_log_prob_pos[valid_rows].mean()


def prototype_separation_loss(prototypes, margin=0.25):
    if prototypes is None or prototypes.ndim != 2 or prototypes.shape[0] < 2:
        if isinstance(prototypes, torch.Tensor):
            return prototypes.new_tensor(0.0)
        return torch.tensor(0.0)

    prototypes = F.normalize(prototypes, dim=1)
    cosine = prototypes @ prototypes.T
    mask = ~torch.eye(cosine.shape[0], dtype=torch.bool, device=cosine.device)
    if not torch.any(mask):
        return prototypes.new_tensor(0.0)
    penalties = F.relu(cosine[mask] - float(margin))
    return penalties.mean() if penalties.numel() > 0 else prototypes.new_tensor(0.0)


def symmetric_kl_divergence(logits_a, logits_b, temperature=1.0):
    if logits_a is None or logits_b is None:
        if isinstance(logits_a, torch.Tensor):
            return logits_a.new_tensor(0.0)
        if isinstance(logits_b, torch.Tensor):
            return logits_b.new_tensor(0.0)
        return torch.tensor(0.0)

    temperature = max(float(temperature), 1e-6)
    log_prob_a = F.log_softmax(logits_a / temperature, dim=1)
    log_prob_b = F.log_softmax(logits_b / temperature, dim=1)
    prob_a = log_prob_a.exp()
    prob_b = log_prob_b.exp()
    loss_ab = F.kl_div(log_prob_a, prob_b, reduction="batchmean")
    loss_ba = F.kl_div(log_prob_b, prob_a, reduction="batchmean")
    return 0.5 * (loss_ab + loss_ba) * (temperature ** 2)

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

    elif loss_name == "balanced_softmax":
        if class_frequencies is None:
            raise ValueError("balanced_softmax requires class_frequencies")
        return BalancedSoftmaxLoss(class_frequencies, label_smoothing=label_smoothing)

    elif loss_name in ["logit_adjusted", "logit_adjusted_cross_entropy"]:
        if class_frequencies is None:
            raise ValueError("logit_adjusted_cross_entropy requires class_frequencies")
        tau = float(config.get("logit_adjust_tau", 1.0))
        return LogitAdjustedCrossEntropyLoss(
            class_frequencies,
            tau=tau,
            label_smoothing=label_smoothing,
        )

    elif loss_name in ["focal", "class_balanced_focal"]:
        gamma = config.get("focal_gamma", 2.0)
        # alpha is only used in focal loss if use_class_weights is enabled
        alpha_weights = weights if config.get("use_class_weights", False) else None
        return FocalLoss(alpha=alpha_weights, gamma=gamma)

    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")
