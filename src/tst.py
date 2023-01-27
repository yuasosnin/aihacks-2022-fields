# A copy of tsai.models.TST but without fastai or tsai dependency.
# The fastai module unfortunately does not work with pytorch_lightning natively.
# Also has renamed arguments and is jittable.

from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.torch_utils.layers import SigmoidRange, Transpose


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        # similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)  # [bs x nhead x q_len x q_len]
        scores = scores / (self.d_k ** 0.5)
        if mask is not None:
            scores.masked_fill_(mask, -1e9)
        attn = F.softmax(scores, dim=-1)  # [bs x nhead x q_len x q_len]
        context = torch.matmul(attn, v)  # [bs x nhead x q_len x d_v]

        return context, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_k: int, d_v: int):
        r"""
        Input shape:  Q, K, V: [batch_size (bs) x q_len x d_model], mask:[q_len x q_len]
        """
        super().__init__()
        self.nhead, self.d_k, self.d_v = nhead, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * nhead, bias=False)
        self.W_K = nn.Linear(d_model, d_k * nhead, bias=False)
        self.W_V = nn.Linear(d_model, d_v * nhead, bias=False)

        self.attention = _ScaledDotProductAttention(self.d_k)

        self.W_O = nn.Linear(nhead * d_v, d_model, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        bs = Q.size(0)

        q_s = self.W_Q(Q).view(bs, self.nhead, -1, self.d_k)  # [bs x nhead x q_len x d_k]
        k_s = self.W_K(K).view(bs, self.nhead, self.d_k, -1)  # [bs x nhead x d_k x q_len]
        v_s = self.W_V(V).view(bs, self.nhead, -1, self.d_v)  # [bs x nhead x q_len x d_v]

        context, attn = self.attention(q_s, k_s, v_s)  # [bs x nhead x q_len x d_v], attn: [bs x nhead x q_len x q_len]
        context = context.transpose(1, 2).contiguous().view(
            bs, -1, self.nhead * self.d_v)  # [bs x q_len x nhead * d_v]
        output = self.W_O(context)  # [bs x q_len x d_model]

        return output, attn


def _get_activation_fn(activation: Union[str, Callable]) -> Callable:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        return activation()


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len: int, d_model: int, nhead: int, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, dropout: float = 0.1, activation: Union[str, Callable] = "gelu"):
        super().__init__()
        assert d_model // nhead, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        d_k = d_model // nhead if d_k is None else d_k
        d_v = d_model // nhead if d_v is None else d_v

        # Multi-Head attention
        self.self_attn = _MultiHeadAttention(d_model, nhead, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        self.batchnorm_attn = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2))

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.batchnorm_ffn = nn.Sequential(
            Transpose(1, 2),
            nn.BatchNorm1d(d_model),
            Transpose(1, 2))

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Multi-Head attention sublayer
        # Multi-Head attention
        src2, attn = self.self_attn(src, src, src, mask=mask)
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_attn(src2)
        src = self.batchnorm_attn(src)

        # Feed-forward sublayer
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_ffn(src2)
        src = self.batchnorm_ffn(src)
        return src


class TSTEncoder(nn.Module):
    def __init__(self, q_len: int, d_model: int, nhead: int, num_layers: int = 1,
                 d_k: Optional[int] = None, d_v: Optional[int] = None, d_ff: Optional[int] = None,
                 dropout: float = 0.1, activation: Union[str, Callable] = "gelu"):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                q_len, d_model, nhead=nhead, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=activation
            ) for i in range(num_layers)])

    def forward(self, src: Tensor) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class TimeSeriesTransformer(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int, max_seq_len: Optional[int] = None,
                 num_layers: int = 3, d_model: int = 128, nhead: int = 16,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,  d_ff: int = 256,
                 dropout: float = 0.1, activation: Union[str, Callable] = "gelu", fc_dropout: float = 0.0,
                 y_range: Optional[tuple] = None, **kwargs):
        r"""
        TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            nhead:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/nhead) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/nhead) = 32.
            d_ff: the dimension of the feedforward network model.
            dropout: amount of residual dropout applied in the encoder.
            activation: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.

        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        super().__init__()
        self.c_out, self.seq_len = c_out, seq_len

        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        # Control temporal resolution
        if max_seq_len is not None and seq_len > max_seq_len: 
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = torch.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(
                nn.ConstantPad1d(padding, value=0.0),
                nn.Conv1d(c_in, d_model, kernel_size=tr_factor, padding=0, stride=tr_factor))
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs)
        else:
            self.W_P = nn.Linear(c_in, d_model)

        # Positional encoding
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            q_len, d_model=d_model, nhead=nhead,
            d_k=d_k, d_v=d_v, d_ff=d_ff,
            dropout=dropout, activation=activation,
            num_layers=num_layers)

        # Head
        self.head_nf = q_len * d_model
        self.head = self.create_head(
            self.head_nf, c_out, activation=activation, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, activation="gelu", fc_dropout=0.0, y_range=None, **kwargs):
        layers = [_get_activation_fn(activation), nn.Flatten()]
        if fc_dropout:
            layers.append(nn.Dropout(fc_dropout))
        layers.append(nn.Linear(nf, c_out))
        if y_range:
            layers.append(SigmoidRange(*y_range))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # x: [bs x nvars x q_len]
        if self.new_q_len:
            u = self.W_P(x).transpose(2, 1)  # [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else:
            u = self.W_P(x.transpose(2, 1))  # [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)  # [bs x q_len x d_model]
        z = z.transpose(2, 1).contiguous()  # [bs x d_model x q_len]

        # Classification / Regression head
        return self.head(z)  # [bs x c_out]
