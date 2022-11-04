import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torchmetrics

from tsai.models.TST import TST as TimeSeriesTransformer
from torchvision.ops import MLP


class StackRNN(pl.LightningModule):
    seq_lens = [70, 139, 18]
    num_classes = 7

    def __init__(self, hidden_size, layers=1, bidirectional=True, dropout=0, fc_layers=1, **hparams):
        super().__init__()
        self.save_hyperparameters()

        # self.rnn1 = nn.GRU(
        #     input_size=1, hidden_size=hidden_size, num_layers=layers, bidirectional=bidirectional,
        #     batch_first=True, dropout=dropout)
        # self.rnn2 = nn.GRU(
        #     input_size=1, hidden_size=hidden_size, num_layers=layers, bidirectional=bidirectional,
        #     batch_first=True, dropout=dropout)
        # self.rnn3 = nn.GRU(
        #     input_size=1, hidden_size=hidden_size, num_layers=layers, bidirectional=bidirectional,
        #     batch_first=True, dropout=dropout)
        
        self.models = nn.ModuleList([nn.GRU(
            input_size=1, hidden_size=hidden_size, num_layers=layers, 
            bidirectional=bidirectional, dropout=dropout,
            batch_first=True
        ) for _ in self.seq_lens])
        
        act = nn.ReLU
        self.act = act()
        
        rnn_out_size = hidden_size*(bidirectional+1)*layers
        self.head = MLP(
            in_channels=rnn_out_size, 
            hidden_channels=[rnn_out_size]*fc_layers + [self.num_classes],
            activation_layer=act)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_recall = torchmetrics.Recall()
        self.valid_recall = torchmetrics.Recall()

    def forward(self, xs):
        bs = xs[0].shape[0]
        hs = [model.forward(x)[1] for model, x in zip(self.models, xs)]
        hs = [h.permute(1,0,2).reshape(bs, -1) for h in hs]
        h = torch.stack(hs, axis=-1)
        h = torch.amax(h, axis=-1)
        return self.head(self.act(h))

    def on_before_batch_transfer(self, batch, dataloader_idx):
        xs, y = batch
        xs = [x.unsqueeze(-1).float() for x in xs]
        y = y.long()
        return xs, y

    def training_step(self, batch, batch_idx):
        xs, y = batch
        output = self.forward(xs)
        loss = self.criterion(output, y)
        self.train_recall(output, y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, y = batch
        output = self.forward(xs)
        loss = self.criterion(output, y)
        self.valid_recall(output, y)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('valid_recall', self.valid_recall, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        xs, _ = batch
        output = self.forward(xs)
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
            activation='relu', 
            **hparams):
        super().__init__()
        self.save_hyperparameters()
        
        # self.model1 = TimeSeriesTransformer(
        #     c_in=1, c_out=d_head, seq_len=self.seq_lens[0],
        #     d_model=d_model, n_heads=nhead, d_ff=dim_feedforward, dropout=dropout, act=activation, n_layers=num_layers)
        # self.model2 = TimeSeriesTransformer(
        #     c_in=1, c_out=d_head, seq_len=self.seq_lens[1],
        #     d_model=d_model, n_heads=nhead, d_ff=dim_feedforward, dropout=dropout, act=activation, n_layers=num_layers)
        # self.model3 = TimeSeriesTransformer(
        #     c_in=1, c_out=d_head, seq_len=self.seq_lens[2],
        #     d_model=d_model, n_heads=nhead, d_ff=dim_feedforward, dropout=dropout, act=activation, n_layers=num_layers)
        
        self.models = nn.ModuleList([TimeSeriesTransformer(
            c_in=1, c_out=d_head, seq_len=seq_len,
            d_model=d_model, n_heads=nhead, d_ff=dim_feedforward, 
            dropout=dropout, act=activation, n_layers=num_layers
        ) for seq_len in self.seq_lens])
        
        if activation == 'relu':
            act = nn.ReLU
        elif activation == 'gelu':
            act = nn.GeLU
        else:
            raise NotImplemented
        self.act = act()
        
        self.head = MLP(
            in_channels=d_head, 
            hidden_channels=[d_head]*num_head_layers + [self.num_classes],
            activation_layer=act)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_recall = torchmetrics.Recall()
        self.valid_recall = torchmetrics.Recall()

    def forward(self, xs):
        hs = [model.forward(x) for model, x in zip(self.models, xs)]
        h = torch.stack(hs, axis=-1)
        h = torch.amax(h, axis=-1)
        return self.head(self.act(h))
    
    # def _prepare_batch(self, batch, train=True):
    #     if train:
    #         xs, y = batch
    #         xs = [x.unsqueeze(1).float() for x in xs]
    #         y = y.long()
    #         return xs, y
    #     else:
    #         xs = batch
    #         xs = [x.unsqueeze(1).float() for x in xs]
    #         return xs
        
    def on_before_batch_transfer(self, batch, dataloader_idx):
        xs, y = batch
        xs = [x.unsqueeze(1).float() for x in xs]
        y = y.long()
        return xs, y
    
    def training_step(self, batch, batch_idx):
        xs, y = batch
        output = self.forward(xs)
        loss = self.criterion(output, y)
        self.train_recall(torch.tensor(output), y)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, y = batch
        output = self.forward(xs)
        loss = self.criterion(output, y)
        self.valid_recall(torch.tensor(output), y)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True)
        self.log('valid_recall', self.valid_recall, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        xs, _ = batch
        output = self.forward(xs)
        return torch.tensor(output)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.gamma)
        return [optimizer], [scheduler]
