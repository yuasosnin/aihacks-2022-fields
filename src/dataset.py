import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


class StackDataset(Dataset):
    def __init__(self, *args, y=None):
        super().__init__()
        self.dfs = args
        self.y = y
        
    def __len__(self):
        return self.dfs[0].shape[0]
    
    def __getitem__(self, idx):
        if self.y is not None:
            return [df.iloc[idx,:].values for df in self.dfs], self.y[idx].astype(int)
        else:
            return [df.iloc[idx,:].values for df in self.dfs], np.nan


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
class StackDataModule(pl.LightningDataModule):
    def __init__(
        self, train_dataset, val_dataset, test_dataset=None, batch_size=64, num_workers=0):
    
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset 
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.g = torch.Generator()
        self.g.manual_seed(1)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True,
                          batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)
