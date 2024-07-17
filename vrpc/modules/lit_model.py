from typing import Dict, List, Tuple, Union
import os

from torchmetrics.functional import accuracy
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch

from lightning.pytorch import LightningModule
from rich import print

from .utils import device_handler


class LitModel(LightningModule):
    """
    A LightningModule wrapper for PyTorch models.

    This class provides a structured way to train, validate, and test PyTorch models
    using PyTorch Lightning, handling the boilerplate code for training loops,
    logging, and device management.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = None,
        optimizer: List[optim.Optimizer] = None,
        scheduler: List[lr_scheduler.LRScheduler] = None,
        checkpoint: str = None,
        num_classes: int = None,
        device: str = "auto",
    ):
        """
        Initialize the Lightning Model.

        Parameters
        ----------
        model : nn.Module
            The neural network model to be trained.
        criterion : nn.Module, optional
            The loss function, by default None
        optimizer : List[optim.Optimizer], optional
            The optimizer, by default None
        scheduler : List[lr_scheduler.LRScheduler], optional
            The learning rate scheduler, by default None
        checkpoint : str, optional
            Path to a checkpoint file for model loading, by default None
        num_classes : int, optional
            The number of output layers, required only when the model does not provide one, by default None
        device : str, optional
            The device to load the model on, by default "auto"
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes or self._get_num_classes()
        if checkpoint:
            self.load(checkpoint, device=device)

    def _get_num_classes(self):
        if not hasattr(self.model, "num_classes"):
            raise AttributeError(
                'The input model does not provide num_classes. "num_classes" parameter cannot be None.'
            )

        return self.model.num_classes

    def _log(
        self, stage: str, loss: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor
    ) -> None:
        """
        Log metrics for a given stage.

        Parameters
        ----------
        stage : str
            The current stage (train, val, or test).
        loss : torch.Tensor
            The loss value.
        y_hat : torch.Tensor
            The model predictions.
        y : torch.Tensor
            The true labels.
        """
        acc = accuracy(
            preds=y_hat, target=y, task="multiclass", num_classes=self.num_classes
        )
        self.log_dict(
            {f"{stage}/loss": loss, f"{stage}/accuracy": acc},
            on_step=False,
            on_epoch=True,
        )

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        """
        Shared step for training, validation, and testing.

        Parameters
        ----------
        batch : tuple
            A tuple containing input data and labels.
        stage : str
            The current stage (train, val, or test).

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self._log(stage=stage, loss=loss, y_hat=y_hat, y=y)
        return loss

    def configure_optimizers(
        self,
    ) -> Union[
        List[optim.Optimizer],
        Tuple[List[optim.Optimizer], List[lr_scheduler.LRScheduler]],
    ]:
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
        Union[
            List[optim.Optimizer],
            Tuple[List[optim.Optimizer], List[lr_scheduler.LRScheduler]],
        ]
            The configured optimizer(s) and scheduler(s).
        """
        return (
            self.optimizer if not self.scheduler else (self.optimizer, self.scheduler)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output of the model.
        """
        return self.model(X)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Perform a training step.

        Parameters
        ----------
        batch : tuple
            A tuple containing input data and labels.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed loss for the training step.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Perform a validation step.

        Parameters
        ----------
        batch : tuple
            A tuple containing input data and labels.
        batch_idx : int
            The index of the current batch.
        """
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Perform a test step.

        Parameters
        ----------
        batch : tuple
            A tuple containing input data and labels.
        batch_idx : int
            The index of the current batch.
        """
        self._shared_step(batch, "test")

    def load(
        self, path: str, strict: bool = True, device: str = "auto", verbose: bool = True
    ) -> None:
        """
        Load a checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        strict : bool, optional
            Whether to strictly enforce that the keys in state_dict match, by default True
        device : str, optional
            The device to load the model on, by default "auto"
        verbose : bool, optional
            Whether to print loading status, by default True
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        if verbose:
            print("[bold]Load checkpoint:[/] Loading...")
        self.load_state_dict(
            torch.load(path, map_location=device_handler(device))["state_dict"],
            strict=strict,
        )
        if verbose:
            print("[bold]Load checkpoint:[/] Done")

    def save_hparams(self, config: Dict) -> None:
        """
        Save hyperparameters.

        Parameters
        ----------
        config : Dict
            Dictionary containing hyperparameters to save.
        """
        self.hparams.update(config)
        self.save_hyperparameters()
