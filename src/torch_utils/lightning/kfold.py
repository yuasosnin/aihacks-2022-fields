# https://github.com/Lightning-AI/lightning/blob/master/examples/pl_loops/kfold.py

from typing import *

# import os.path
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from sklearn.model_selection import KFold

import pytorch_lightning as pl
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class BaseKFoldDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


class KFoldLoop(Loop):
    def __init__(self, ensemble_model: pl.LightningModule, num_folds: int, 
                 checkpoint_type: str = 'last') -> None:
        super().__init__()
        self.ensemble_model = ensemble_model
        self.num_folds = num_folds
        self.current_fold: int = 0
        assert checkpoint_type in {'best', 'last'}
        self._checkpoint_type = checkpoint_type
        self.checkpoint_paths = []

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        if self.trainer.checkpoint_callback is None:
            raise MisconfigurationException("Checkpointer has to be specified to run KFoldLoop")
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())
        self.checkpoint_callback_state_dict = deepcopy(self.trainer.checkpoint_callback.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.

        # the test loop normally expects the model to be the pure LightningModule, but since we are running the
        # test loop during fitting, we need to temporarily unpack the wrapped module
        wrapped_model = self.trainer.strategy.model
        self.trainer.strategy.model = self.trainer.strategy.lightning_module
        self.trainer.test_loop.run()
        self.trainer.strategy.model = wrapped_model
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        # self.trainer.save_checkpoint(os.path.join(self.export_path, f"model.{self.current_fold}.pt"))
        if self._checkpoint_type == 'best':
            assert self.trainer.checkpoint_callback.best_model_path
            self.checkpoint_paths.append(self.trainer.checkpoint_callback.best_model_path)
        elif self._checkpoint_type == 'last':
            assert self.trainer.checkpoint_callback.last_model_path
            self.checkpoint_paths.append(self.trainer.checkpoint_callback.last_model_path)
        
        # restore the original weights + optimizers and schedulers + primary checkpoint
        self.trainer.lightning_module.load_state_dict(deepcopy(self.lightning_module_state_dict))
        self.trainer.checkpoint_callback.load_state_dict(deepcopy(self.checkpoint_callback_state_dict))
        
        old_state = self.trainer.state.fn  # HACK
        self.trainer.state.fn = TrainerFn.FITTING  
        self.trainer.strategy.setup_optimizers(self.trainer)  # https://github.com/Lightning-AI/lightning/issues/12409
        self.trainer.state.fn = old_state
        
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        # checkpoint_paths = [os.path.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        ensemble_model = self.ensemble_model(type(self.trainer.lightning_module), self.checkpoint_paths)
        ensemble_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(ensemble_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
