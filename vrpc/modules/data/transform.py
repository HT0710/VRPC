from typing import Tuple

import torchvision.transforms as T
from torch import Tensor

from modules.utils import tuple_handler


class DataTransformation:
    """Contains Transformation methods for image"""

    @staticmethod
    def DEFAULT(image_size: Tuple | list | int = (224, 224)) -> Tensor:
        """
        Apply a default set of image transformations to the input image.

        Args:
            image_size (Tuple | list | int, optional): The desired size of the output image. Defaults to (224, 224).

        Returns:
            Tensor: The transformed image as a PyTorch tensor.
        """
        return T.Compose(
            [
                T.Resize(tuple_handler(image_size, max_dim=2), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def TOPIL(image_size: Tuple | list | int = (224, 224)) -> Tensor:
        """
        Convert a PyTorch tensor to a PIL image and apply a series of image transformations.

        Args:
            image_size (Tuple | list | int, optional): The desired size of the output image. Defaults to (224, 224).

        Returns:
            Tensor: The transformed image as a PyTorch tensor.
        """
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize(tuple_handler(image_size, max_dim=2), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def AUGMENT_LV0(image_size: Tuple | list | int = (224, 224)) -> Tensor:
        """Femboy 🫵

        Apply a default set of image transformations to the input image.

        Args:
            image_size (Tuple | list | int, optional): The desired size of the output image. Defaults to (224, 224).

        Returns:
            Tensor: The transformed image as a PyTorch tensor.
        """
        return T.Compose(
            [
                T.Resize(tuple_handler(image_size, max_dim=2), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def AUGMENT_LV1(image_size: Tuple | list | int = (224, 224)) -> Tensor:
        """Survivor 🌟

        Apply a common set of image transformations to the input image.

        Args:
            image_size (Tuple | list | int, optional): The desired size of the output image. Defaults to (224, 224).

        Returns:
            Tensor: The transformed image as a PyTorch tensor.
        """
        return T.Compose(
            [
                T.RandomResizedCrop(
                    tuple_handler(image_size, max_dim=2), antialias=True
                ),
                T.RandomHorizontalFlip(p=0.1),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def AUGMENT_LV2(image_size: Tuple | list | int = (224, 224)) -> Tensor:
        """Trooper 🔫

        Apply a medium set of image transformations to the input image.

        Args:
            image_size (Tuple | list | int, optional): The desired size of the output image. Defaults to (224, 224).

        Returns:
            Tensor: The transformed image as a PyTorch tensor.
        """
        return T.Compose(
            [
                T.RandomResizedCrop(
                    tuple_handler(image_size, max_dim=2), antialias=True
                ),
                T.RandomHorizontalFlip(p=0.2),
                T.RandomRotation(30),
                T.GaussianBlur(3),
                T.RandomPerspective(0.2, p=0.2),
                T.RandomAutocontrast(p=0.2),
                T.RandomAdjustSharpness(2, p=0.2),
                T.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.075
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def AUGMENT_LV3(image_size: Tuple | list | int = (224, 224)) -> Tensor:
        """Veterant 🚀

        Apply a big set of image transformations to the input image.

        Args:
            image_size (Tuple | list | int, optional): The desired size of the output image. Defaults to (224, 224).

        Returns:
            Tensor: The transformed image as a PyTorch tensor.
        """
        return T.Compose(
            [
                T.RandomResizedCrop(
                    tuple_handler(image_size, max_dim=2), antialias=True
                ),
                T.RandomHorizontalFlip(p=0.35),
                T.RandomRotation(60),
                T.GaussianBlur(5),
                T.RandomPerspective(0.35, p=0.35),
                T.RandomAutocontrast(p=0.35),
                T.RandomAdjustSharpness(4, p=0.35),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.RandomEqualize(p=0.35),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def AUGMENT_LV4(image_size: Tuple | list | int = (224, 224)) -> Tensor:
        """Hellraiser 🗿

        Apply a huge set of image transformations to the input image.

        Args:
            image_size (Tuple | list | int, optional): The desired size of the output image. Defaults to (224, 224).

        Returns:
            Tensor: The transformed image as a PyTorch tensor.
        """
        return T.Compose(
            [
                T.RandomResizedCrop(
                    tuple_handler(image_size, max_dim=2), antialias=True
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(90),
                T.GaussianBlur(7),
                T.RandomPerspective(0.5, p=0.5),
                T.RandomAutocontrast(p=0.5),
                T.RandomAdjustSharpness(6, p=0.5),
                T.RandomAffine(degrees=30, translate=(0.1, 0.1)),
                T.RandomEqualize(p=0.5),
                T.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def AUGMENT_LV5(image_size: Tuple | list | int = (224, 224)) -> Tensor:
        """Doom Slayer 💀

        Apply a massive set of image transformations to the input image.

        Args:
            image_size (Tuple | list | int, optional): The desired size of the output image. Defaults to (224, 224).

        Returns:
            Tensor: The transformed image as a PyTorch tensor.
        """
        return T.Compose(
            [
                T.RandomResizedCrop(
                    tuple_handler(image_size, max_dim=2), antialias=True
                ),
                T.RandomHorizontalFlip(p=0.69),
                T.RandomRotation(96),
                T.GaussianBlur(9.6),
                T.RandomPerspective(0.69, p=0.69),
                T.RandomAutocontrast(p=0.69),
                T.RandomAdjustSharpness(9.6, p=0.69),
                T.RandomAffine(degrees=69, translate=(0.6, 0.9)),
                T.RandomEqualize(p=0.69),
                T.ColorJitter(brightness=0.69, contrast=0.69, saturation=0.69, hue=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
