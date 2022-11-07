from typing import *
import copy
import pytorch_lightning as pl


class PrintMetricsCallback(pl.callbacks.Callback):
    """
    A Callback to print metrics into console.
    """

    def __init__(self, metrics: Optional[Sequence[str]] = None) -> None:
        self.epoch = -1
        self.metrics = metrics

    def on_validation_epoch_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics_dict = copy.copy(trainer.callback_metrics)
        metrics = self.metrics if self.metrics is not None else metrics_dict.keys()
        print('epoch:', self.epoch)
        for metric in metrics:
            if metric in metrics_dict:
                print(f'{metric}:', metrics_dict[metric].item())
        print('-' * 80)
        self.epoch += 1
