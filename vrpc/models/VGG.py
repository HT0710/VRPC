from typing import List, Optional

from torch import nn
import torch

from modules.utils import yaml_handler


class VGG(nn.Module):
    """
    VGG (Visual Geometry Group) neural network implementation.

    This class implements the VGG architecture, which is a convolutional neural network
    known for its simplicity and depth. It consists of a series of convolutional layers
    followed by fully connected layers.
    """

    def __init__(
        self,
        version: str,
        num_classes: int,
        hidden_dim: Optional[int] = 4096,
        in_channels: Optional[int] = 3,
        classifier: Optional[nn.Module] = None,
    ):
        """
        Initialize the VGG network.

        Parameters
        ----------
        version : str
            VGG version ["11", "13", "16", "19"]
        num_classes : int
            The number of output classes.
        hidden_dim : int, optional
            The number of hidden dimensions, by default 4096
        in_channels : int, optional
            The number of input channels, by default 3
        classifier : nn.Module, optional
            Custom classifier module, by default None
        """
        super().__init__()
        _config = yaml_handler(path="vrpc/configs/models/vgg.yaml")
        self.features = self._build_layers(_config[version], in_channels)
        self.classifier = classifier or self._build_default_classifier(
            num_classes, hidden_dim
        )
        self.num_classes = num_classes
        self.version = version

    def _build_layers(self, config: List[int], in_channels: int) -> nn.Sequential:
        """
        Build the convolutional layers of the network.

        Parameters
        ----------
        config : List[int]
            Configuration list for the convolutional layers.
        in_channels : int
            Number of input channels.

        Returns
        -------
        nn.Sequential
            A sequential container of the built layers.
        """
        layers = []
        for x in config:
            if x == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers.extend([conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)])
                in_channels = x
        return nn.Sequential(*layers)

    def _build_default_classifier(
        self, num_classes: int, hidden_dim: int = 4090
    ) -> nn.Sequential:
        """
        Build the default classifier for the VGG network.

        Parameters
        ----------
        num_classes : int
            The number of output classes.

        Returns
        -------
        nn.Sequential
            The default classifier module.
        """
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the VGG network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
