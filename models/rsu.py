import torch
import torch.nn as nn
from .blocks import ConvBNReLU

# U2Net-inspired Residual U-blocks (RSU)

def _upsample_like(src, tar):
    return torch.nn.functional.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

class RSU7(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU7, self).__init__()
        self.c0 = ConvBNReLU(in_ch, out_ch, dirate=1)
        
        self.c1 = ConvBNReLU(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.c2 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.c3 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.c4 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.c5 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.c6 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        
        self.c7 = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        
        self.d6 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d5 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d4 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d3 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d2 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d1 = ConvBNReLU(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = self.c0(x)
        
        h1 = self.c1(hx)
        h = self.pool1(h1)
        
        h2 = self.c2(h)
        h = self.pool2(h2)
        
        h3 = self.c3(h)
        h = self.pool3(h3)
        
        h4 = self.c4(h)
        h = self.pool4(h4)
        
        h5 = self.c5(h)
        h = self.pool5(h5)
        
        h6 = self.c6(h)
        h7 = self.c7(h6)
        
        # Decoder
        d6 = self.d6(torch.cat((h7, h6), 1))
        
        d6_up = _upsample_like(d6, h5)
        d5 = self.d5(torch.cat((d6_up, h5), 1))
        
        d5_up = _upsample_like(d5, h4)
        d4 = self.d4(torch.cat((d5_up, h4), 1))
        
        d4_up = _upsample_like(d4, h3)
        d3 = self.d3(torch.cat((d4_up, h3), 1))
        
        d3_up = _upsample_like(d3, h2)
        d2 = self.d2(torch.cat((d3_up, h2), 1))
        
        d2_up = _upsample_like(d2, h1)
        d1 = self.d1(torch.cat((d2_up, h1), 1))
        
        return d1 + hx


class RSU6(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU6, self).__init__()
        self.c0 = ConvBNReLU(in_ch, out_ch, dirate=1)
        self.c1 = ConvBNReLU(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c2 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c3 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c4 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c5 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.c6 = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        
        self.d5 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d4 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d3 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d2 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d1 = ConvBNReLU(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = self.c0(x)
        h1 = self.c1(hx)
        h = self.pool1(h1)
        h2 = self.c2(h)
        h = self.pool2(h2)
        h3 = self.c3(h)
        h = self.pool3(h3)
        h4 = self.c4(h)
        h = self.pool4(h4)
        h5 = self.c5(h)
        h6 = self.c6(h5)
        
        d5 = self.d5(torch.cat((h6, h5), 1))
        d5_up = _upsample_like(d5, h4)
        d4 = self.d4(torch.cat((d5_up, h4), 1))
        d4_up = _upsample_like(d4, h3)
        d3 = self.d3(torch.cat((d4_up, h3), 1))
        d3_up = _upsample_like(d3, h2)
        d2 = self.d2(torch.cat((d3_up, h2), 1))
        d2_up = _upsample_like(d2, h1)
        d1 = self.d1(torch.cat((d2_up, h1), 1))
        
        return d1 + hx


class RSU5(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU5, self).__init__()
        self.c0 = ConvBNReLU(in_ch, out_ch, dirate=1)
        self.c1 = ConvBNReLU(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c2 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c3 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c4 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.c5 = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        
        self.d4 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d3 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d2 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d1 = ConvBNReLU(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = self.c0(x)
        h1 = self.c1(hx)
        h = self.pool1(h1)
        h2 = self.c2(h)
        h = self.pool2(h2)
        h3 = self.c3(h)
        h = self.pool3(h3)
        h4 = self.c4(h)
        h5 = self.c5(h4)
        
        d4 = self.d4(torch.cat((h5, h4), 1))
        d4_up = _upsample_like(d4, h3)
        d3 = self.d3(torch.cat((d4_up, h3), 1))
        d3_up = _upsample_like(d3, h2)
        d2 = self.d2(torch.cat((d3_up, h2), 1))
        d2_up = _upsample_like(d2, h1)
        d1 = self.d1(torch.cat((d2_up, h1), 1))
        
        return d1 + hx


class RSU4(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU4, self).__init__()
        self.c0 = ConvBNReLU(in_ch, out_ch, dirate=1)
        self.c1 = ConvBNReLU(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c2 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.c3 = ConvBNReLU(mid_ch, mid_ch, dirate=1)
        self.c4 = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        
        self.d3 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d2 = ConvBNReLU(mid_ch*2, mid_ch, dirate=1)
        self.d1 = ConvBNReLU(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = self.c0(x)
        h1 = self.c1(hx)
        h = self.pool1(h1)
        h2 = self.c2(h)
        h = self.pool2(h2)
        h3 = self.c3(h)
        h4 = self.c4(h3)
        
        d3 = self.d3(torch.cat((h4, h3), 1))
        d3_up = _upsample_like(d3, h2)
        d2 = self.d2(torch.cat((d3_up, h2), 1))
        d2_up = _upsample_like(d2, h1)
        d1 = self.d1(torch.cat((d2_up, h1), 1))
        
        return d1 + hx


class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU4F, self).__init__()
        self.c0 = ConvBNReLU(in_ch, out_ch, dirate=1)
        self.c1 = ConvBNReLU(out_ch, mid_ch, dirate=1)
        self.c2 = ConvBNReLU(mid_ch, mid_ch, dirate=2)
        self.c3 = ConvBNReLU(mid_ch, mid_ch, dirate=4)
        self.c4 = ConvBNReLU(mid_ch, mid_ch, dirate=8)
        
        self.d3 = ConvBNReLU(mid_ch*2, mid_ch, dirate=4)
        self.d2 = ConvBNReLU(mid_ch*2, mid_ch, dirate=2)
        self.d1 = ConvBNReLU(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = self.c0(x)
        h1 = self.c1(hx)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        
        d3 = self.d3(torch.cat((h4, h3), 1))
        d2 = self.d2(torch.cat((d3, h2), 1))
        d1 = self.d1(torch.cat((d2, h1), 1))
        
        return d1 + hx
