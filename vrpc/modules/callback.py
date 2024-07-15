from typing import List
from rich import print
import yaml

from lightning.pytorch import Trainer, LightningModule
import lightning.pytorch.callbacks as cb


class PrintTrainResult(cb.Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch

        current_lr = ", ".join(
            [f"{optim.param_groups[0]['lr']:.2e}" for optim in trainer.optimizers]
        )

        results = trainer.callback_metrics
        result_train = (
            f"loss: {results['train/loss']:.4f}, acc: {results['train/accuracy']:.3f}"
        )
        result_val = (
            f"loss: {results['val/loss']:.4f}, acc: {results['val/accuracy']:.3f}"
        )

        print(
            f"[bold]Epoch[/]( {epoch} ) "
            f"[bold]Lr[/]( {current_lr} ) "
            f"[bold]Train[/]({result_train}) "
            f"[bold]Val[/]({result_val})"
        )


def custom_callbacks() -> List[cb.Callback]:
    """
    Configuration for custom callbacks

    Returns
    -------
    List
        A list of defined callbacks.
    """
    with open("vrpc/configs/callbacks.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    callbacks = []
    if cfg["verbose"]:
        callbacks.append(PrintTrainResult())
    if cfg["model_summary"]:
        callbacks.append(cb.RichModelSummary())
    if cfg["progress_bar"]:
        callbacks.append(cb.RichProgressBar())
    if cfg["lr_monitor"]:
        callbacks.append(cb.LearningRateMonitor("epoch"))
    if cfg["enable_checkpoint"]:
        callbacks.append(cb.ModelCheckpoint(**cfg["checkpoint"]))
    if cfg["enable_early_stopping"]:
        callbacks.append(cb.EarlyStopping(**cfg["early_stopping"]))
    return callbacks
