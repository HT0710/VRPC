from typing import List, Union
from rich import print

from lightning.pytorch import Trainer, LightningModule
import lightning.pytorch.callbacks as cb

from .utils import yaml_handler


class PrintTrainResult(cb.Callback):
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

        if value > prev_value:
            trend = f"[{'green' if up_green else 'red'}]▲[/]"
        elif value < prev_value:
            trend = f"[{'red' if up_green else 'green'}]▼[/]"
        else:
            trend = "[grey]-[/]"

        # Format the value
        formatted_value = f"{value:{format_spec}}"

        return f"{trend} {formatted_value}"

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
