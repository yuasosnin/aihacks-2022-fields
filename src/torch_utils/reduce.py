from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxReduce(nn.Module):
    """
    Reduces out given dimension of a tensor by taking max across its values.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
            
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.amax(h, axis=self.dim)
    

class AvgReduce(nn.Module):
    """
    Reduces out given dimension of a tensor by taking average across its values.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
            
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.mean(h, axis=self.dim)


class ParamReduce(nn.Module):
    """
    Reduces out given dimension of a tensor 
    by taking weighted combination across its values. 
    Weights add up to 1.
    """
    def __init__(self, in_dim: int, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self._coeffs = nn.Parameter(torch.Tensor(in_dim).normal_(1/in_dim, 0.1))
    
    @property
    def coeffs(self) -> torch.Tensor:
        return F.softmax(self._coeffs)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sum(h*self.coeffs, axis=self.dim)
