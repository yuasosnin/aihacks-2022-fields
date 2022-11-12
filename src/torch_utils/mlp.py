# https://pytorch.org/vision/stable/_modules/torchvision/ops/misc.html#MLP

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        layers = []
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        if norm_layer is not None:
            layers.append(norm_layer(out_features))
        if activation is not None:
            layers.append(activation())
        layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class MLP(nn.Sequential):
    """
    Implements the multi-layer perceptron (MLP) module.
    A copy of torchvision.ops.MLP, but with renamed arguments, 
    no dropout after last layer, and additional ``act_first`` argument.

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
        bias (bool): Whether to use bias in the linear layer. Default: ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
        act_first (bool): Wheather to apply activation, norm and dropout 
            before first linear layer. Default: ``False``
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        act_first: bool = False,
    ):
        layers = []
        in_dim = in_features
        
        if act_first:
            if norm_layer is not None:
                layers.append(norm_layer(in_features))
            if activation is not None:
                layers.append(activation())
            layers.append(nn.Dropout(dropout))
        
        for hidden_dim in hidden_features[:-1]:
            block = MLPBlock(
                in_dim, hidden_dim, 
                activation=activation, 
                norm_layer=norm_layer, 
                dropout=dropout,
                bias=bias)
            layers.append(block)
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, hidden_features[-1], bias=bias))
        
        super().__init__(*layers)
