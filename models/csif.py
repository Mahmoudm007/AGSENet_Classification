import torch
import torch.nn as nn
import torch.nn.functional as F

class SCIP(nn.Module):
    """
    Spatial Context Information Perception (SCIP).
    Lightweight spatial attention to enhance pixel-level contextual encoding.
    Uses conv -> norm -> activation -> conv(1) -> sigmoid to produce spatial attention.
    """
    def __init__(self, in_channels):
        super(SCIP, self).__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        att_map = self.spatial_att(x)
        return x * att_map

class CSII(nn.Module):
    """
    Channel Significant Information Interaction (CSII).
    Uses self-attention over channels to improve saliency-aware channel modeling.
    """
    def __init__(self, in_channels):
        super(CSII, self).__init__()
        self.conv_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        
        # [B, C, H, W]
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        
        # Global Average Pooling -> [B, C, 1, 1] -> [B, C, 1] and [B, 1, C]
        q_tilde = F.adaptive_avg_pool2d(q, 1).view(B, C, 1)
        k_tilde = F.adaptive_avg_pool2d(k, 1).view(B, C, 1).transpose(1, 2)
        
        # Channel Affinity: A = softmax(q @ k) -> [B, C, C]
        attn = torch.bmm(q_tilde, k_tilde)  # [B, C, C]
        attn = F.softmax(attn, dim=-1)
        
        # Reshape V -> [B, C, H*W]
        v_flat = v.view(B, C, -1)
        
        # P = A @ V -> [B, C, H*W]
        out = torch.bmm(attn, v_flat)
        out = out.view(B, C, H, W)
        
        # Residual fusion
        return x + self.alpha * out

class CSIF(nn.Module):
    """
    Channel Saliency Information Focus (CSIF).
    Composed of SCIP (spatial) and CSII (channel).
    """
    def __init__(self, in_channels):
        super(CSIF, self).__init__()
        self.scip = SCIP(in_channels)
        self.csii = CSII(in_channels)

    def forward(self, x):
        x = self.scip(x)
        x = self.csii(x)
        return x
