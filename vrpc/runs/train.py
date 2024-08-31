import shutil

from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torchvision import models, datasets
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as ls
import torch.optim as optim
import torch.nn as nn
import torch

from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything, Trainer, LightningDataModule

from omegaconf import DictConfig, open_dict
from torchinfo import summary
from rich import traceback
import rootutils
import hydra

rootutils.autosetup()
traceback.install()

from torchview import draw_graph
from modules import LitModel, scheduler_with_warmup, custom_callbacks
from modules.data import CustomDataModule, DataTransformation as DT
from modules.data.tinyimagenetloader import (
    TrainTinyImageNetDataset,
    TestTinyImageNetDataset,
    id_dict as classes,
)


from models.Custom import BasicStage
from models.convnext import convnext_tiny


class LitDataModule(LightningDataModule):
    def __init__(self, image_size, batch_size, num_workers):
        super().__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def classes(self):
        return classes

    def train_dataloader(self):
        return DataLoader(
            TrainTinyImageNetDataset(
                transform=DT.AUGMENT_LV1(image_size=self.image_size)
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=1,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TestTinyImageNetDataset(
                transform=DT.AUGMENT_LV0(image_size=self.image_size)
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=1,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs since we already have lightning logs
    shutil.rmtree("outputs")

    # Set precision
    torch.set_float32_matmul_precision("high")

    # Set seed
    if cfg["set_seed"]:
        seed_everything(seed=cfg["set_seed"], workers=True)

    # Define dataset
    dataset = LitDataModule(
        image_size=64, batch_size=cfg["trainer"]["batch_size"], num_workers=18
    )
    # trainset = datasets.CIFAR10(
    #     root="vrpc/data",
    #     train=True,
    #     download=True,
    #     transform=DT.AUGMENT_LV1(image_size=64),
    # )
    # trainloader = DataLoader(
    #     trainset, batch_size=cfg["trainer"]["batch_size"], shuffle=True, num_workers=18
    # )

    # testset = datasets.CIFAR10(
    #     root="vrpc/data",
    #     train=False,
    #     download=True,
    #     transform=DT.AUGMENT_LV0(image_size=64),
    # )
    # testloader = DataLoader(
    #     testset, batch_size=cfg["trainer"]["batch_size"], shuffle=False, num_workers=18
    # )

    # dataset = CustomDataModule(
    #     **cfg["data"],
    #     batch_size=cfg["trainer"]["batch_size"],
    #     num_workers=cfg["num_workers"] if torch.cuda.is_available() else 0,
    # )

    # Define model
    model = BasicStage(num_classes=len(dataset.classes), image_size=64)
    # model = convnext_tiny(num_classes=len(dataset.classes))
    # model = models.vit_b_16(num_classes=len(dataset.classes), dropout=0.1)

    # show = draw_graph(
    #     model,
    #     input_size=(512, 3, 64, 64),
    #     device="meta",
    #     roll=True,
    #     save_graph=True,
    #     expand_nested=True,
    # )
    # show.visual_graph
    # exit()

    # summary(model, (1, 3, 64, 64))
    # exit()

    # Setup loss
    loss = nn.CrossEntropyLoss()

    # Setup optimizer
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=cfg["trainer"]["learning_rate"],
        weight_decay=cfg["trainer"]["learning_rate"],
    )

    # Setup scheduler
    scheduler = scheduler_with_warmup(
        scheduler=ls.CosineAnnealingLR(
            optimizer=optimizer, T_max=cfg["trainer"]["num_epoch"]
        ),
        warmup_epochs=cfg["scheduler"]["warmup_epochs"],
        start_factor=cfg["scheduler"]["start_factor"],
    )

    # Lightning model
    lit_model = LitModel(
        model=model,
        criterion=loss,
        optimizer=[optimizer],
        scheduler=[scheduler],
        checkpoint=cfg["trainer"]["checkpoint"],
        device="auto",
        num_classes=len(dataset.classes),
    )

    # Save config
    with open_dict(cfg):
        cfg["model"]["name"] = model._get_name()
        if hasattr(model, "version"):
            cfg["model"]["version"] = model.version
    lit_model.save_hparams(cfg)

    # Lightning trainer
    trainer = Trainer(
        max_epochs=cfg["trainer"]["num_epoch"],
        precision=cfg["trainer"]["precision"],
        logger=TensorBoardLogger(save_dir="."),
        callbacks=custom_callbacks(),
        gradient_clip_val=0.5,
    )

    # Lightning tuner
    # tuner = Tuner(trainer)

    # Auto-scale batch size by growing it exponentially
    # tuner.scale_batch_size(lit_model, datamodule=dataset)

    # Training
    trainer.fit(lit_model, dataset)

    # Testing
    # trainer.test(lit_model, dataset)


if __name__ == "__main__":
    main()
