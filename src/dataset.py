from typing import *

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
from sklearn.model_selection import KFold

from .torch_utils.lightning import BaseKFoldDataModule


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class StackKFoldDataModule(BaseKFoldDataModule):    
    def __init__(
        self, train_dataset, pred_dataset, const, batch_size=64, num_workers=0, seed=5):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = train_dataset
        self.pred_dataset = pred_dataset
        self.const = const
        self.seed = seed
        self.g = torch.Generator()
        self.g.manual_seed(seed)
        
    @staticmethod
    def _get_lens(n, sizes):
        rest = [int(n*s) for s in sizes[1:]]
        train = n - sum(rest)
        return [train] + rest
    
    def _calc_ts_stats(self, train_dataset):
        n_ds = len(self.dataset.tensors) - 1 - self.const
        means = [torch.stack([x[i].mean(axis=(1)) for x in train_dataset]).mean(axis=0)[None,:,None] for i in range(n_ds)]
        stds = [torch.stack([x[i].std(axis=(1)) for x in train_dataset]).mean(axis=0)[None,:,None] for i in range(n_ds)]
        return means, stds
    
    def _calc_const_stats(self, train_dataset):
        const_idx = len(self.dataset.tensors) - 2
        mean = torch.stack([x[const_idx] for x in train_dataset]).mean(0)
        std = torch.stack([x[const_idx] for x in train_dataset]).std(0)
        return mean, std
    
    def _normalize(self):
        y_tensor = self.dataset.tensors[-1]
        if self.const:
            train_const_tensor = self.dataset.tensors[-2]
            train_tensors = self.dataset.tensors[:-2]
            pred_const_tensor = self.pred_dataset.tensors[-1]
            pred_tensors = self.pred_dataset.tensors[:-1]
        else:
            train_tensors = self.dataset.tensors[:-1]
            pred_tensors = self.pred_dataset.tensors
        means, stds = self._calc_ts_stats(self.train_dataset)
        
        train_tensors = [(ds-means[i])/stds[i] for i, ds in enumerate(train_tensors)]# + [y_tensor]
        pred_tensors = [(ds-means[i])/stds[i] for i, ds in enumerate(pred_tensors)]

        if self.const:
            mean, std = self._calc_const_stats(self.train_dataset)
            train_const_tensor = (train_const_tensor - mean) / std
            pred_const_tensor = (pred_const_tensor - mean) / std
            train_tensors.append(train_const_tensor)
            pred_tensors.append(pred_const_tensor)
        train_tensors.append(y_tensor)
        
        self.dataset = TensorDataset(*train_tensors)
        self.pred_dataset = TensorDataset(*pred_tensors)
        
    def setup(self, stage: Optional[str] = None):
        # split into train and test
        self.g.manual_seed(self.seed)
        self.train_dataset, self.test_dataset = random_split(
            self.dataset, self._get_lens(len(self.dataset), [0.8, 0.2]), generator=self.g)
        # normalize based on test
        self._normalize()
        # split again with normalized tansors
        self.g.manual_seed(self.seed)
        self.train_dataset, self.test_dataset = random_split(
            self.dataset, self._get_lens(len(self.dataset), [0.8, 0.2]), generator=self.g)
        # split initially into train and val to use outside of KFoldLoop
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
