import shutil

from torchvision import models
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as ls
import torch.optim as optim
import torch.nn as nn
import torch

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything, Trainer

from omegaconf import DictConfig, open_dict
from torchinfo import summary
from rich import traceback
import rootutils
import hydra

rootutils.autosetup()
traceback.install()

from modules import LitModel, scheduler_with_warmup, custom_callbacks
from modules.data import CustomDataModule, DataTransformation as DT
from modules.data.tinyimagenetloader import (
    TrainTinyImageNetDataset,
    TestTinyImageNetDataset,
    id_dict,
)

from models.Custom import BasicStage
from models.CoAtNet import CoAtNet


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
    trainloader = DataLoader(
        TrainTinyImageNetDataset(transform=DT.AUGMENT_LV0(image_size=64)),
        batch_size=cfg["trainer"]["batch_size"],
        num_workers=18,
        shuffle=True,
    )
    testloader = DataLoader(
        TestTinyImageNetDataset(transform=DT.AUGMENT_LV0(image_size=64)),
        batch_size=cfg["trainer"]["batch_size"],
        num_workers=18,
        shuffle=False,
    )
    # dataset = CustomDataModule(
    #     **cfg["data"],
    #     batch_size=cfg["trainer"]["batch_size"],
    #     num_workers=cfg["num_workers"] if torch.cuda.is_available() else 0,
    # )

    # Define model
    # model = CoAtNet(in_ch=3, image_size=64, num_classes=len(id_dict))
    model = BasicStage(num_classes=len(id_dict))
    # model = models.VisionTransformer(
    #     patch_size=16,
    #     num_layers=4,
    #     num_heads=4,
    #     hidden_dim=256,
    #     mlp_dim=256,
    #     image_size=224,
    #     num_classes=len(id_dict),
    # )

    # summary(model, (1, 3, 64, 64))
    # exit()

    # Setup loss
    loss = nn.CrossEntropyLoss()

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
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
        num_classes=len(id_dict),
        device="auto",
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
    )

    # Training
    trainer.fit(lit_model, trainloader, testloader)

    # Testing
    # trainer.test(lit_model, dataset)


if __name__ == "__main__":
    main()
