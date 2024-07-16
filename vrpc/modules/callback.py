from typing import List
from rich import print

from lightning.pytorch import Trainer, LightningModule
import lightning.pytorch.callbacks as cb

from .utils import yaml_handler


class PrintTrainResult(cb.Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        results = trainer.callback_metrics

        current_lr = ", ".join(
            f"{optim.param_groups[0]['lr']:.2e}" for optim in trainer.optimizers
        )

        train_result = (
            f"loss: {results['train/loss']:.4f}, acc: {results['train/accuracy']:.3f}"
        )

        output = [
            f"[bold]Epoch[/]( {epoch} )",
            f"[bold]Lr[/]( {current_lr} )",
            f"[bold]Train[/]({train_result})",
        ]

        if "val/loss" in results:
            val_result = (
                f"loss: {results['val/loss']:.4f}, acc: {results['val/accuracy']:.3f}"
            )
            output.append(f"[bold]Val[/]({val_result})")

        print(" ".join(output))


def custom_callbacks() -> List[cb.Callback]:
    """
    Configure and return a list of custom callbacks for PyTorch Lightning.

    Returns
    -------
    List[cb.Callback]
        A list of configured PyTorch Lightning callback objects.
    """
    cfg = yaml_handler("vrpc/configs/callbacks.yaml")
    callbacks = []

    callback_map = {
        "verbose": PrintTrainResult(),
        "model_summary": cb.RichModelSummary(),
        "progress_bar": cb.RichProgressBar(),
        "lr_monitor": cb.LearningRateMonitor("epoch"),
        "enable_checkpoint": cb.ModelCheckpoint(**cfg["checkpoint"]),
        "enable_early_stopping": cb.EarlyStopping(**cfg["early_stopping"]),
    }

    for key, callback in callback_map.items():
        if cfg.get(key, False):
            callbacks.append(callback)

    return callbacks
