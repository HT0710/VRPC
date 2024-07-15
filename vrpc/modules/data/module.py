from typing import List, Tuple
from pathlib import Path
import os

from torch.utils.data import DataLoader, ConcatDataset
from lightning.pytorch import LightningDataModule
from torchvision.datasets import ImageFolder

from rich import print

from modules.data import DataTransformation
from modules.utils import workers_handler


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        augment_level: int = 0,
        image_size: Tuple[int, int] | list = (224, 224),
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """
        Custom Data Module for PyTorch Lightning

        Args:
            data_path (str): Path to the dataset.
            batch_size (int, optional): Batch size for data loading. Default: 32
            augment_level (int, optional): Augmentation level for data transformations. Default: 0
            image_size (tuple, optional): Size of the input images. Default: (224, 224)
            num_workers (int, optional): Number of data loading workers. Default: 0
            pin_memory (bool, optional): Whether to pin memory for faster data transfer. Default: True
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.augment_level = augment_level
        self.image_size = image_size
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": workers_handler(num_workers),
            "pin_memory": pin_memory,
        }

    @property
    def classes(self) -> List[str]:
        return sorted(os.listdir(self.data_path / "train"))

    def prepare_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(str(self.data_path))

    def setup(self, stage: str):
        if not hasattr(self, "dataset"):
            transform_lv = {
                i: getattr(DataTransformation, f"AUGMENT_LV{i}") for i in range(6)
            }

            if self.augment_level not in transform_lv:
                raise ValueError(
                    "Use 0 for the default transformation, or scale up to 5 for the strongest effect."
                )

            data_sets = []

            self.train_set = ImageFolder(
                (self.data_path / "train"),
                transform=transform_lv[self.augment_level](self.image_size),
            )

            data_sets.append(self.train_set)

            if (self.data_path / "val").exists():
                self.val_set = ImageFolder(
                    (self.data_path / "val"),
                    transform=transform_lv[0](self.image_size),
                )
                data_sets.append(self.val_set)

            if (self.data_path / "test").exists():
                self.test_set = ImageFolder(
                    (self.data_path / "test"),
                    transform=transform_lv[0](self.image_size),
                )
                data_sets.append(self.test_set)

            self.dataset = ConcatDataset(data_sets)

        if stage == "fit":
            print(f"[bold]Data path:[/] [green]{self.data_path}[/]")
            print(f"[bold]Number of data:[/] {len(self.dataset):,}")
            print(f"[bold]Number of classes:[/] {len(self.classes):,}")

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.loader_config, shuffle=True)

    def val_dataloader(self):
        return (
            DataLoader(dataset=self.val_set, **self.loader_config, shuffle=False)
            if hasattr(self, "val_set")
            else None
        )

    def test_dataloader(self):
        return (
            DataLoader(dataset=self.test_set, **self.loader_config, shuffle=False)
            if hasattr(self, "test_set")
            else None
        )
