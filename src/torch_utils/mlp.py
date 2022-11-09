# https://pytorch.org/vision/stable/_modules/torchvision/ops/misc.html#MLP

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    """
    This block implements the multi-layer perceptron (MLP) module.
    A copy of torchvision.ops.MLP, but with renamed arguments 
    and no dropout after last layer.

    Args:
        in_features (int): Number of features of the input
        hidden_features (List[int]): List of the hidden features dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): 
            Norm layer that will be stacked on top of linear layer. 
            If ``None`` this layer wont be used. 
            Default: ``None``
        activation (Callable[..., torch.nn.Module], optional): 
            Activation function which will be stacked on top 
            of the normalization layer (if not None), 
            otherwise on top of linear layer. 
            If ``None`` this layer wont be used. 
            Default: ``torch.nn.ReLU``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        layers = []
        in_dim = in_features
        
        for hidden_dim in hidden_features[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            if activation is not None:
                layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_features[-1], bias=bias))

        super().__init__(*layers)
