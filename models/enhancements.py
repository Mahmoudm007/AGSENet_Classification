import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECABlock(nn.Module):
    """Efficient Channel Attention for lightweight channel recalibration."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        t = int(abs((math.log2(channels) + b) / gamma)) if channels > 1 else 1
        kernel_size = max(3, t if t % 2 else t + 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.activation(y)
        return x * y.expand_as(x)


class LearnableGeMPool2d(nn.Module):
    """Generalized mean pooling with learnable exponent."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=8.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x


class InputRepresentationEnhancer(nn.Module):
    """Additive shallow enhancer to preserve lane texture and edge detail."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(in_ch)
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.context_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
        )
        self.texture_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch * 3, in_ch, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.pre_norm(x)
        local = self.local_branch(x_norm)
        context = self.context_branch(x_norm)
        texture = self.texture_branch(x_norm)
        gate = self.gate(torch.cat([local, context, texture], dim=1))
        enriched = local + context + texture
        return x + self.residual_scale * gate * enriched


class ScaleAttentionFusion(nn.Module):
    """Self-attention over pooled multi-scale tokens."""

    def __init__(self, token_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = self._resolve_num_heads(token_dim, num_heads)
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.token_score = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )

    @staticmethod
    def _resolve_num_heads(token_dim: int, requested_heads: int) -> int:
        requested_heads = max(1, int(requested_heads))
        for heads in range(min(token_dim, requested_heads), 0, -1):
            if token_dim % heads == 0:
                return heads
        return 1

    def forward(self, tokens: torch.Tensor):
        norm_tokens = self.norm1(tokens)
        attn_out, attn_weights = self.attn(
            norm_tokens,
            norm_tokens,
            norm_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        refined = tokens + attn_out
        refined = refined + self.ffn(self.norm2(refined))
        token_weights = torch.softmax(self.token_score(refined).squeeze(-1), dim=1)
        pooled = torch.sum(refined * token_weights.unsqueeze(-1), dim=1)
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)
        return pooled, refined, token_weights, attn_weights


class PrototypeRelationRefiner(nn.Module):
    """Refine class prototypes with lightweight self-attention."""

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.query = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value = nn.Linear(feature_dim, feature_dim, bias=False)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = feature_dim ** -0.5

    def forward(self, prototypes: torch.Tensor):
        x = self.norm1(prototypes)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        relation = torch.softmax((query @ key.transpose(0, 1)) * self.scale, dim=-1)
        relational = relation @ value
        refined = prototypes + self.dropout(relational)
        refined = refined + self.ffn(self.norm2(refined))
        return refined, relation


class DynamicFusionGate(nn.Module):
    """Image/text gate that adapts mixing strength per sample."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        logit_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = visual_dim + text_dim + logit_dim + 4
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        image_embed: torch.Tensor,
        description_logits: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(description_logits, dim=1)
        topk = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        top1 = topk[:, :1]
        margin = top1 - topk[:, 1:2] if probs.shape[1] > 1 else top1
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        logit_std = description_logits.std(dim=1, keepdim=True, unbiased=False)
        gate_input = torch.cat(
            [visual_features, image_embed, description_logits, entropy, top1, margin, logit_std],
            dim=1,
        )
        return self.net(gate_input)


class EnhancementClassifier(nn.Module):
    """Residual classifier head on top of richer pooled features."""

    def __init__(self, in_features: int, hidden_features: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

