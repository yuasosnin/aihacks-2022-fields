import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
import torchmetrics

from tsai.all import *
from torchvision.ops import MLP


class StackRNN(pl.LightningModule):
    def __init__(self, hidden_size, layers=1, bidirectional=True, dropout=0, fc_layers=1, **hparams):
        super().__init__()
        self.save_hyperparameters()

        self.rnn1 = nn.GRU(
            input_size=1, hidden_size=hidden_size, num_layers=layers, bidirectional=bidirectional,
            batch_first=True, dropout=dropout)
        self.rnn2 = nn.GRU(
            input_size=1, hidden_size=hidden_size, num_layers=layers, bidirectional=bidirectional,
            batch_first=True, dropout=dropout)
        self.rnn3 = nn.GRU(
            input_size=1, hidden_size=hidden_size, num_layers=layers, bidirectional=bidirectional,
            batch_first=True, dropout=dropout)
        
        self.act = nn.ReLU()
        
        rnn_out_size = hidden_size*(bidirectional+1)*layers
        self.head = MLP(
            in_channels=rnn_out_size, 
            hidden_channels=[rnn_out_size]*fc_layers + [7],
            activation_layer=nn.ReLU)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_recall = torchmetrics.Recall()
        self.valid_recall = torchmetrics.Recall()
        
    def forward(self, x1, x2, x3):
        bs = x1.shape[0]
        _, h1 = self.rnn1(x1)
        _, h2 = self.rnn2(x2)
        _, h3 = self.rnn3(x3)

        h1 = h1.permute(1,0,2).reshape(bs, -1)
        h2 = h2.permute(1,0,2).reshape(bs, -1)
        h3 = h3.permute(1,0,2).reshape(bs, -1)

        h = torch.stack((h1,h2,h3), axis=-1)
        h = torch.amax(h, axis=-1)
        # h = h1
        return self.head(self.act(h))
    
    def _prepare_batch(self, batch, train=True):
        if train:
            (x1, x2, x3), y = batch
            x1 = x1.unsqueeze(-1).float()
            x2 = x2.unsqueeze(-1).float()
            x3 = x3.unsqueeze(-1).float()
            y = y.long()
            return (x1, x2, x3), y
        else:
            (x1, x2, x3) = batch
            x1 = x1.unsqueeze(-1).float()
            x2 = x2.unsqueeze(-1).float()
            x3 = x3.unsqueeze(-1).float()
            return (x1, x2, x3)

    def training_step(self, batch, batch_idx):
        (x1, x2, x3), y = self._prepare_batch(batch)
        output = self.forward(x1, x2, x3)
        loss = self.criterion(output, y)
        self.train_recall(output, y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x1, x2, x3), y = self._prepare_batch(batch)
        output = self.forward(x1, x2, x3)
        loss = self.criterion(output, y)
        self.valid_recall(output, y)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('valid_recall', self.valid_recall, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        (x1, x2, x3) = self._prepare_batch(batch, train=False)
        output = self.forward(x1, x2, x3)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.gamma)
        return [optimizer], [scheduler]


class StackTransformer(pl.LightningModule):
    seq_lens = [70, 139, 18]
    num_classes = 7
    
    def __init__(
            self, 
            d_model=64, 
            nhead=1, 
            dim_feedforward=64, 
            d_head=64, 
            num_layers=1, 
            num_head_layers=1,
            dropout=0, 
            activation="relu", 
            **hparams):
        super().__init__()
        self.save_hyperparameters()
        
        self.model1 = TST(
            c_in=1, c_out=d_head, seq_len=self.seq_lens[0],
            d_model=d_model, n_heads=nhead, d_ff=dim_feedforward, dropout=dropout, act=activation, n_layers=num_layers)
        self.model2 = TST(
            c_in=1, c_out=d_head, seq_len=self.seq_lens[1],
            d_model=d_model, n_heads=nhead, d_ff=dim_feedforward, dropout=dropout, act=activation, n_layers=num_layers)
        self.model3 = TST(
            c_in=1, c_out=d_head, seq_len=self.seq_lens[2],
            d_model=d_model, n_heads=nhead, d_ff=dim_feedforward, dropout=dropout, act=activation, n_layers=num_layers)
        
        self.act = nn.ReLU()
        
        self.head = MLP(
            in_channels=d_head, 
            hidden_channels=[d_head]*num_head_layers + [self.num_classes],
            activation_layer=nn.ReLU)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_recall = torchmetrics.Recall()
        self.valid_recall = torchmetrics.Recall()
        
    def forward(self, x1, x2, x3):
        h1 = self.model1.forward(x1)
        h2 = self.model2.forward(x2)
        h3 = self.model3.forward(x3)

        h = torch.stack((h1,h2,h3), axis=-1)
        h = torch.amax(h, axis=-1)
        # h = h1
        return self.head(self.act(h))
    
    def _prepare_batch(self, batch, train=True):
        if train:
            (x1, x2, x3), y = batch
            x1 = x1.unsqueeze(1).float()
            x2 = x2.unsqueeze(1).float()
            x3 = x3.unsqueeze(1).float()
            y = y.long()
            return (x1, x2, x3), y
        else:
            (x1, x2, x3) = batch
            x1 = x1.unsqueeze(1).float()
            x2 = x2.unsqueeze(1).float()
            x3 = x3.unsqueeze(1).float()
            return (x1, x2, x3)
    
    def training_step(self, batch, batch_idx):
        (x1, x2, x3), y = self._prepare_batch(batch)
        output = self.forward(x1, x2, x3)
        loss = self.criterion(output, y)
        self.train_recall(torch.tensor(output), y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (x1, x2, x3), y = self._prepare_batch(batch)
        output = self.forward(x1, x2, x3)
        loss = self.criterion(output, y)
        self.valid_recall(torch.tensor(output), y)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('valid_recall', self.valid_recall, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        (x1, x2, x3) = self._prepare_batch(batch, train=False)
        output = self.forward(x1, x2, x3)
        return torch.tensor(output)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.gamma)
        return [optimizer], [scheduler]
