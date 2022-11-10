from typing import *

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader, random_split
import pytorch_lightning as pl
from sklearn.model_selection import KFold

from .torch_utils.lightning import BaseKFoldDataModule


class StackDataset(Dataset):
    def __init__(self, *args, static=None, y=None):
        super().__init__()
        # self.dfs = [df.values for df in args]
        dfs = []
        for df in args:
            if isinstance(df, pd.DataFrame):
                df = df.values
            if len(df.shape) == 2:
                df = df[:,None,:]
            if static is not None:
                static_ = np.repeat(static[:,:,None], df.shape[2], axis=2)
                dfs.append(np.concatenate([df, static_], axis=1))
            else:
                dfs.append(df)
        self.dfs = dfs
        if isinstance(y, pd.Series):
            y = y.values
        self.y = y if y is not None else None
        
    def __len__(self):
        return self.dfs[0].shape[0]
    
    def __getitem__(self, idx):
        xs = [df[idx,:,:] for df in self.dfs]
        y = self.y[idx] if self.y is not None else np.nan
        
        return xs, y


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class StackKFoldDataModule(BaseKFoldDataModule):    
    def __init__(
        self, train_dataframes, pred_dataframes, train_y, train_static=None, pred_static=None, batch_size=64, num_workers=0, seed=5):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataframes = train_dataframes
        self.train_static = train_static
        self.train_y = train_y
        self.pred_dataframes = pred_dataframes
        self.pred_static = pred_static
        self.g = torch.Generator()
        self.g.manual_seed(seed)
        
    @staticmethod
    def _get_lens(n, sizes):
        rest = [int(n*s) for s in sizes[1:]]
        train = n - sum(rest)
        return [train] + rest
        
    def setup(self, stage: Optional[str] = None):
        dataset = StackDataset(*self.train_dataframes, static=self.train_static, y=self.train_y)
        self.pred_dataset = StackDataset(*self.pred_dataframes, static=self.pred_static)
        self.train_dataset, self.test_dataset = random_split(
            dataset, self._get_lens(len(dataset), [0.8, 0.2]), generator=self.g)
        self.train_fold, self.val_fold = random_split(
            self.train_dataset, self._get_lens(len(self.train_dataset), [0.9, 0.1]), generator=self.g)
    

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)
    

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, shuffle=True,
                          batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.pred_dataset, shuffle=False,
                          batch_size=self.batch_size, num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)


class StackDataModule(StackKFoldDataModule):
    def __init__(self, train_dataframes, pred_dataframes, train_y, batch_size=64, num_workers=0):
        super().__init__(
            train_dataframes, pred_dataframes, train_y, batch_size=batch_size, num_workers=num_workers)

    def setup_folds(self, *args, **kwargs):
        raise NotImplemented
        
    def setup_fold_index(self, *args, **kwargs):
        raise NotImplemented
