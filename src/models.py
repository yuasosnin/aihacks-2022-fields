from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RAdam
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl
import torchmetrics

from tsai.models.TST import TST as TimeSeriesTransformer
# from torchvision.ops import MLP
from .torch_utils import MLP
from .torch_utils import MaxReduce, AvgReduce, ParamReduce


class StackTransformer(pl.LightningModule):
    num_classes = 7
    
    def __init__(
            self, 
            c_ins=[1,1,1,1],
            seq_lens=[70, 139, 18, 55],
            d_model=64, 
            nhead=1, 
            dim_feedforward=64, 
            d_head=64, 
            num_layers=1, 
            num_head_layers=1,
            dropout=0, 
            fc_dropout=0,
            activation=nn.ReLU,
            reduction='avg',
            **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.seq_lens = seq_lens
        
        self.models = nn.ModuleList([TimeSeriesTransformer(
            c_in=c_in, c_out=d_head, seq_len=seq_len,
            d_model=d_model, n_heads=nhead, d_ff=dim_feedforward, 
            dropout=dropout, act=activation, n_layers=num_layers,
            fc_dropout=dropout
        ) for seq_len, c_in in zip(self.seq_lens, c_ins)])
        
        self.pool = self._get_pool(reduction)
        # self.act = activation()
        
        d_head_in = d_head*(len(seq_lens)) if reduction == 'flatten' else d_head
        self.head = MLP(
            in_features=d_head_in, 
            hidden_features=[d_head]*num_head_layers + [self.num_classes],
            activation=activation, dropout=fc_dropout, act_first=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_recall = torchmetrics.Recall()
        self.valid_recall = torchmetrics.Recall()
        self.test_recall = torchmetrics.Recall()
    
    def _get_pool(self, reduction):
        if reduction == 'flatten':
            return nn.Flatten(start_dim=-2, end_dim=-1)
        elif reduction == 'max':
            return MaxReduce(dim=-1)
        elif reduction == 'avg':
            return AvgReduce(dim=-1)
        elif reduction == 'sum':
            return SumReduce(dim=-1)
        elif reduction == 'param':
            return ParamReduce(in_dim=len(self.seq_lens))
        else:
            raise ValueError
    
    
    def forward(self, xs):
        hs = [model.forward(x) for model, x in zip(self.models, xs)]
        h = torch.stack(hs, axis=-1)
        h = self.pool(h)
        return self.head(h)
    
    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     xs, y = batch
    #     xs = [x.float() for x in xs]
    #     y = y.long()
    #     return xs, y
    
    def training_step(self, batch, batch_idx):
        xs, y = batch[:-1], batch[-1]
        output = self.forward(xs)
        loss = self.criterion(output, y)
        # wrap into torch.tensor for compatiability with fastai
        self.train_recall(torch.tensor(output), y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xs, y = batch[:-1], batch[-1]
        output = self.forward(xs)
        loss = self.criterion(output, y)
        self.valid_recall(torch.tensor(output), y)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('valid_recall', self.valid_recall, on_step=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        xs, y = batch[:-1], batch[-1]
        output = self.forward(xs)
        loss = self.criterion(output, y)
        self.test_recall(torch.tensor(output), y)
        self.log('test_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=True, on_epoch=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        xs = batch
        output = self.forward(xs)
        return torch.tensor(output)
    
    def configure_optimizers(self):
        optimizer = RAdam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = ExponentialLR(
            optimizer, self.hparams.gamma)
        return [optimizer], [scheduler]


class EnsembleVotingModel(pl.LightningModule):
    def __init__(self, model_cls: pl.LightningModule, checkpoint_paths: List[str]) -> None:
        super().__init__()
        self.models = nn.ModuleList(
            [model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        
        self.criterion = nn.CrossEntropyLoss()
        self.test_recall = torchmetrics.Recall()
        
    def forward(self, xs):
        outputs = [model(xs) for model in self.models]
        return torch.stack(outputs).mean(0)
    
    # doesnt work for some reason
    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     xs, y = batch
    #     xs = [x.float() for x in xs]
    #     y = y.long()
    #     return xs, y
    
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        xs, y = batch[:-1], batch[-1]
        output = self.forward(xs)
        loss = self.criterion(output, y)
        self.test_recall(torch.tensor(output), y)
        # can't be called 'test_loss' and 'test_recall'!
        self.log('test_full_loss', loss.item())
        self.log('test_full_recall', self.test_recall)
        return loss
    
    def predict_step(self, batch, batch_idx):
        xs = batch
        xs = [x.float() for x in xs]
        output = self.forward(xs)
        return torch.tensor(output)
