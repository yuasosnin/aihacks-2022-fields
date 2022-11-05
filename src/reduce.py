import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxReduce(nn.Module):
    """
    Reduces out given dimension of a tensor by taking max across its values.
    """
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
            
    def forward(self, h):
        return torch.amax(h, axis=self.axis)
    

class AvgReduce(nn.Module):
    """
    Reduces out given dimension of a tensor by taking average across its values.
    """
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
            
    def forward(self, h):
        return torch.mean(h, axis=self.axis)


class ParamReduce(nn.Module):
    """
    Reduces out given dimension of a tensor 
    by taking weighted combination across its values. 
    Weights add up to 1.
    """
    def __init__(self, in_dim, axis=-1):
        super().__init__()
        self.axis = axis
        self.in_dim = in_dim
        self._coeffs = nn.Parameter(torch.Tensor(in_dim).normal_(1/in_dim, 0.1))
    
    @property
    def coeffs(self):
        return F.softmax(self._coeffs)
        
    def forward(self, h):
        return torch.sum(h*self.coeffs, axis=self.axis)
