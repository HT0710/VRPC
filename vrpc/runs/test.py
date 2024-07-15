import yaml

from rich import traceback
import rootutils

rootutils.autosetup()
traceback.install()

from models.VIT import VisionTransformer  # noqa: E402
from models.Custom_VGG import VGG  # noqa: E402


def main():

    model = VGG("11", num_classes=10)
    # model = VisionTransformer(
    #     image_size=224,
    #     patch_size=16,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_dim=768,
    #     mlp_dim=3072,
    # )

    print(model)


if __name__ == "__main__":
    main()
