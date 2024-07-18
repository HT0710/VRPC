import os

from lightning.pytorch import seed_everything, Trainer
from torch.nn import CosineSimilarity
import torch
import cv2

from torchinfo import summary
from rich import traceback
import rootutils


rootutils.autosetup()
traceback.install()

from modules.data.transform import DataTransformation  # noqa: E402
from modules.data.module import CustomDataModule  # noqa: E402
from modules import LitModel  # noqa: E402
from models import VGG  # noqa: E402
from models.Custom import BasicBlock  # noqa: E402


def main():
    model = BasicBlock(200)

    lit_model = LitModel(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        checkpoint="lightning_logs/version_131/checkpoints/last.ckpt",
        num_classes=200,
    ).eval()

    # lit_model.model.classifier = lit_model.model.classifier[:-3]

    dataset = CustomDataModule(
        data_path="vrpc/data/rpc-classify_x",
        data_limit=None,
        batch_size=1,
        image_size=672,
    )
    dataset.setup("predict")
    loader = dataset.test_set

    result = {}

    with torch.inference_mode():
        for X, y in loader:
            out = lit_model(X.unsqueeze(0))

            out = torch.softmax(out, 1)

            pred = torch.argmax(out, 1)

            if pred == y:
                print(pred)
                print(torch.sort(out, 1).values)
                print(dataset.classes[pred])

                print(torch.max(out, 1)[0])
                exit()


if __name__ == "__main__":
    main()
