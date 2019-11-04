from config import ConfigArgs as args
import torch
import torch.nn as nn
import numpy as np

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope=0.2, inplace=True):
        super(LeakyReLU, self).__init__(negative_slope=negative_slope, inplace=inplace)

class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same',
                    dilation=1, groups=1, lrelu=True, weight_norm=True):
        super(CustomConv1d, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2 * dilation
            self.even_kernel = not bool(kernel_size % 2)
        
        self.lrelu = nn.LeakyReLU(0.2) if lrelu else None
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups
        )
        if weight_norm:
            nn.utils.weight_norm(self.conv)

    def forward(self, x):
        y = self.lrelu(x) if self.lrelu is not None else x
        y = self.conv(y)
        y = y[:, :, :-1] if self.even_kernel else y
        return y