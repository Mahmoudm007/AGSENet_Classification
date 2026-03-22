import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBNReLU

def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv(x)

class SSIE(nn.Module):
    """
    Spatial Saliency Information Exploration (SSIE).
    Takes high-level (Fh) and low-level (Fl) features, rejects noise,
    refines edges via spatial attention, and optionally fuses them.
    In classification context, it acts as a very strong top-down feature refining neck.
    """
    def __init__(self, in_channels_h, in_channels_l, out_channels):
        super(SSIE, self).__init__()
        
        # Project high-level features if channels don't match low-level for element-wise ops
        self.proj_h = nn.Conv2d(in_channels_h, in_channels_l, 1) if in_channels_h != in_channels_l else nn.Identity()
        
        self.sa_n = SpatialAttention(in_channels_l)
        self.sa_e = SpatialAttention(in_channels_l)
        
        self.fusion_conv = ConvBNReLU(in_channels_l * 2, out_channels, dirate=1)

    def forward(self, Fh, Fl):
        # Align spatial dimensions
        if Fh.shape[2:] != Fl.shape[2:]:
            Fh = _upsample_like(Fh, Fl)
            
        Fh = self.proj_h(Fh)
        
        # Noise rejection branch
        Fn = Fh - Fl
        
        # Edge refinement branch
        Fe = Fh * Fl
        
        # Spatial self-attention
        sa_n_map = self.sa_n(Fn)  # [B, 1, H, W]
        sa_e_map = self.sa_e(Fe)  # [B, 1, H, W]
        
        # Reweight
        Fh_n = sa_n_map * Fh
        Fl_n = sa_n_map * Fl
        
        Fh_e = sa_e_map * Fh
        Fl_e = sa_e_map * Fl
        
        # Aggregate
        Fh_prime = Fh_n + Fh_e
        Fl_prime = Fl_n + Fl_e
        
        # Fuse
        Ff = torch.cat((Fh_prime, Fl_prime), dim=1) # [B, 2*in_channels_l, H, W]
        
        # Reduce
        out = self.fusion_conv(Ff)
        return out
