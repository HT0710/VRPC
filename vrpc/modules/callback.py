from datetime import datetime
from typing import Any, List, Union, Mapping

from lightning.pytorch import Trainer, LightningModule
import lightning.pytorch.callbacks as cb

from rich import print
from torch import Tensor

from .utils import yaml_handler


class PrintResult(cb.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.prev = {}

    def _format_with_trend(
        self,
        name: str,
        value: Union[int, float],
        format_spec: str = "",
        up_green: bool = True,
    ) -> str:
        # Store the previous value and get the trend
        prev_value = self.prev.get(name, value)
        self.prev[name] = value

        # Check current vs previous
        if value > prev_value:
            trend = "green" if up_green else "red"
        elif value < prev_value:
            trend = "red" if up_green else "green"
        else:
            trend = "grey"

        # Format the value
        formatted_value = f"{value:{format_spec}}"

        return f"[{trend}]{formatted_value}[/]"

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        lr_sequence = ",".join(f"lr{i}" for i, _ in enumerate(trainer.optimizers))

        with open(f"{trainer.logger.log_dir}/results.csv", "a") as f:
            f.write(f"Epoch,{lr_sequence},train_loss,train_acc,val_loss,val_acc\n")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        results = trainer.callback_metrics

        lr = [
            self._format_with_trend(f"lr{i}", optim.param_groups[0]["lr"], ".1e", False)
            for i, optim in enumerate(trainer.optimizers)
        ]

        train_result = [
            f"loss: {self._format_with_trend('train_loss', results['train/loss'], '.4f', False)}",
            f"acc: {self._format_with_trend('train_acc', results['train/accuracy'], '.3f', True)}",
        ]

        output = [
            f"[bold]Epoch[/]( {epoch} )",
            f"[bold]Lr[/]( {', '.join(lr)} )",
            f"[bold]Train[/]({', '.join(train_result)})",
        ]

        if "val/loss" in results:
            val_result = [
                f"loss: {self._format_with_trend('val_loss', results['val/loss'], '.4f', False)}",
                f"acc: {self._format_with_trend('val_acc', results['val/accuracy'], '.3f', True)}",
            ]
            output.append(f"[bold]Val[/]({', '.join(val_result)})")

        print(" ".join(output))

        with open(f"{trainer.logger.log_dir}/results.csv", "a") as f:
            lr_values = ",".join(
                f"{optim.param_groups[0]['lr']:.2e}" for optim in trainer.optimizers
            )
            f.write(
                f"{epoch},{lr_values},"
                f"{results['train/loss']:.5f},{results['train/accuracy']:.4f},"
                f"{results['val/loss']:.5f},{results['val/accuracy']:.4f}\n"
            )


def custom_callbacks() -> List[cb.Callback]:
    """
    Configure and return a list of custom callbacks for PyTorch Lightning.

    Returns
    -------
    List[cb.Callback]
        A list of configured PyTorch Lightning callback objects.
    """
    cfg = yaml_handler("vrpc/configs/callbacks.yaml")

    callback_map = {
        "verbose": PrintResult(),
        "progress_bar": cb.RichProgressBar(),
        "lr_monitor": cb.LearningRateMonitor("epoch"),
        "enable_checkpoint": cb.ModelCheckpoint(**cfg["checkpoint"]),
        "enable_early_stopping": cb.EarlyStopping(**cfg["early_stopping"]),
    }

    return [callback for key, callback in callback_map.items() if cfg.get(key, False)]
