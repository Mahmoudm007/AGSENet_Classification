import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    """
    Standard Convolution + BatchNorm + ReLU block used extensively in AGSENet (RSU blocks).
    """
    def __init__(self, in_ch: int, out_ch: int, dirate: int = 1):
        super(ConvBNReLU, self).__init__()
        # Use dilation rate for both padding and dilation to keep spatial dimensions constant
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dirate, dilation=dirate, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
