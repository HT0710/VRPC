from typing import List, Optional

from torch import nn
import torch

from modules.utils import yaml_handler


class BasicConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Scaled Dot-Product Attention mechanism.

        Parameters
        ----------
        embed_dim : int
            The dimension of the model (input and output dimension).
        num_heads : int
            The number of attention heads.
        dropout : float, optional
            Dropout probability (default is 0.1).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of the Scaled Dot-Product Attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch_size, seq_len, embed_dim).
        key : torch.Tensor
            Key tensor of shape (batch_size, seq_len, embed_dim).
        value : torch.Tensor
            Value tensor of shape (batch_size, seq_len, embed_dim).
        mask : torch.Tensor, optional
            Mask tensor of shape (batch_size, seq_len, seq_len).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        query = self._reshape_to_batches(self.q_proj(query))
        key = self._reshape_to_batches(self.k_proj(key))
        value = self._reshape_to_batches(self.v_proj(value))

        scaling_factor = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, value)
        output = self._reshape_from_batches(output)
        return self.out_proj(output)

    def _reshape_to_batches(self, x):
        """
        Reshape the input tensor to separate attention heads.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Reshaped tensor of shape (batch_size, num_heads, seq_len, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def _reshape_from_batches(self, x):
        """
        Reshape the tensor back to original shape after attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_heads, seq_len, head_dim).

        Returns
        -------
        torch.Tensor
            Reshaped tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)


class BasicStage(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.avgpool = nn.AvgPool2d(24)
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.maxpool1d = nn.MaxPool1d(2, stride=2)
        self.conv1 = BasicConvBlock(in_features=3, out_features=32)
        self.conv2 = BasicConvBlock(in_features=32, out_features=64)
        self.att = ScaledDotProductAttention(embed_dim=64, num_heads=8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 24, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.maxpool2d(x)
        x = self.conv2(x)
        x = self.maxpool2d(x)
        x = x.reshape(x.shape[0], 64, -1).transpose(1, 2)
        x = self.att(x, x, x)
        x = self.maxpool1d(x.transpose(1, 2))
        x = self.fc(x)
        return x


class Custom(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: Optional[int] = 4096,
        in_channels: Optional[int] = 3,
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
        """
        super().__init__()
        _config = yaml_handler(path="vrpc/configs/models/custom.yaml")
        self.features = self._build_layers(_config["11"], in_channels)
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

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
