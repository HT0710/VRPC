from typing import List, Optional, Tuple, Union
from pathlib import Path
import os

from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from lightning.pytorch import LightningDataModule
from torchvision.datasets import ImageFolder

from rich.table import Table
from rich import print

from modules.data import DataTransformation
from modules.utils import workers_handler


class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        augment_level: int = 0,
        image_size: Union[Tuple[int, int], List] = (224, 224),
        data_limit: Optional[float] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """
        Custom Data Module for PyTorch Lightning

        Args:
            data_path (str): Path to the dataset.
            batch_size (int, optional): Batch size for data loading. Default: 32
            augment_level (int, optional): Augmentation level for data transformations (0 -> 5). Default: 0
            image_size (tuple, optional): Size of the input images. Default: (224, 224)
            data_limit (float, optional): Limit for the size of the dataset (0 -> 1.0). Default: None
            num_workers (int, optional): Number of data loading workers. Default: 0
            pin_memory (bool, optional): Whether to pin memory for faster data transfer. Default: True
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.augment_level = augment_level
        self.image_size = image_size
        self.data_limit = self._check_limit(data_limit)
        self.loader_config = {
            "batch_size": batch_size,
            "num_workers": workers_handler(num_workers),
            "pin_memory": pin_memory,
        }

    @property
    def classes(self) -> List[str]:
        """Get the list of class names."""
        return sorted(os.listdir(self.data_path / "train"))

    @staticmethod
    def _check_limit(value: Optional[float]) -> Optional[float]:
        return 1 if not value or (1 < value < 0) else value

    def _limit_data(self, data: Dataset) -> Subset:
        return random_split(
            dataset=data, lengths=(self.data_limit, 1 - self.data_limit)
        )[0]

    def _summary(self) -> None:
        table = Table(title="[bold]Sets Distribution[/]")
        table.add_column("Set", style="cyan", no_wrap=True)
        table.add_column("Total", justify="right", style="magenta")
        table.add_column("Split", justify="right", style="green")
        for set_name, set_len in [
            ("Train", len(self.train_set)),
            ("Val", len(self.val_set)),
            ("Test", len(self.test_set)),
        ]:
            table.add_row(set_name, f"{set_len:,}", f"{set_len/len(self.dataset):.0%}")
        print(table)
        output = [
            (
                f"[bold]Number of data:[/] {len(self.dataset):,}"
                + f" ([red]{self.data_limit:.0%}[/])"
                if self.data_limit != 1
                else ""
            ),
            f"[bold]Number of classes:[/] {len(self.classes):,}",
            f"[bold]Data path:[/] [green]{self.data_path}[/]",
        ]
        print("\n".join(output))

    def prepare_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def setup(self, stage: str):
        if not hasattr(self, "dataset"):
            transform_lv = {
                i: getattr(DataTransformation, f"AUGMENT_LV{i}") for i in range(6)
            }

            if self.augment_level not in transform_lv:
                raise ValueError(
                    "Use 0 for zero transformation, or scale up to 5 for the strongest effect."
                )

            self.train_set = ImageFolder(
                (self.data_path / "train"),
                transform=transform_lv[self.augment_level](self.image_size),
            )

            self.val_set = ImageFolder(
                (self.data_path / "val"),
                transform=transform_lv[0](self.image_size),
                allow_empty=True,
            )

            self.test_set = ImageFolder(
                (self.data_path / "test"),
                transform=transform_lv[0](self.image_size),
                allow_empty=True,
            )

            if self.data_limit:
                for data_set in ["train_set", "val_set", "test_set"]:
                    setattr(
                        self, data_set, self._limit_data(data=getattr(self, data_set))
                    )

            self.dataset = ConcatDataset([self.train_set, self.val_set, self.test_set])

        if stage == "fit":
            self._summary()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.loader_config, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.loader_config, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.loader_config, shuffle=False)
