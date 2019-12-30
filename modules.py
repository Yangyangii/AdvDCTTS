from config import ConfigArgs as args
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as norm
import numpy as np
import layers as ll

class ResidualBlock1d(nn.Module):
    def __init__(self, in_planes, planes):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = ll.CustomConv1d(in_planes, planes, 3, lrelu=True)
        self.conv2 = ll.CustomConv1d(planes, planes, 3, lrelu=True)
        self.proj = nn.Conv1d(in_planes, planes, 1) if in_planes != planes else None
    
    def forward(self, x):
        identity = x
        
        y = self.conv1(x)
        y = self.conv2(y)

        if self.proj is not None:
            identity = self.proj(identity)
        y = y + identity
        return y
