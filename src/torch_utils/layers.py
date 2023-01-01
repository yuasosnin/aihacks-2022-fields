from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x): 
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()
        else: 
            return x.transpose(*self.dims)

    def __repr__(self): 
        if self.contiguous: 
            return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else: 
            return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x):
        return torch.sigmoid(x) * (self.high - self.low) + self.low

    def __repr__(self):
        return f"{self.__class__.__name__}({self.low}, {self.high})"
