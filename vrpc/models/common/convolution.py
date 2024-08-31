from torch.nn import functional as F
import torch.nn as nn
import torch

from .attention import ChannelAttention


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, num_channels, height, width)

    return x


def dim_shuffle(x: torch.Tensor, dim: int) -> torch.Tensor:
    permutation = torch.randperm(x.size(dim), device=x.device)

    x = x.index_select(dim, permutation)

    return x


def drop_path(x: torch.Tensor, keep_prob: float = 1.0, inplace: bool = False):
    if keep_prob == 1.0 or not x.requires_grad:
        return x

    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)

    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x


class BasicConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int = 1,
        bias: bool = False,
        normalize: bool = True,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, stride, (kernel - 1) // 2, bias=bias
        )
        self.norm = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class CustomConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        _hidden_dims = in_channels * 2

        self.conv1 = BasicConvBlock(in_channels, _hidden_dims, 3, activation=None)
        self.conv2 = nn.Conv2d(_hidden_dims, _hidden_dims, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(_hidden_dims, _hidden_dims, 3, padding=1, bias=False)
        self.conv4 = BasicConvBlock(_hidden_dims, out_channels, 3, activation=None)

        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(out_channels)
        self.c_att = ChannelAttention(out_channels, reduce_ratio=16, dropout=0.1)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = y = self.conv1(x)
        x = self.act(x)
        x = z = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x + y)
        x = self.act(x)
        x = self.conv4(x + y + z)
        x = self.act(x + shortcut)
        x = self.c_att(x)
        x = self.norm(x)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 7, padding=3, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, 1, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels * expansion, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.GELU()

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x += self.shortcut(identity)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.act(out)

        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction=4):
        super().__init__()
        hidden_dim = out_channels // reduction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.GELU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += self.shortcut(identity)
        x = self.act(x)
        return x


class InvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = in_channels == out_channels and stride == 1

        layers = []
        if expansion != 1:
            # Expansion
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.GELU(),
                ]
            )

        # Depthwise
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
            ]
        )

        # Projection
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        ksize,
        input_filters,
        output_filters,
        expand_ratio=1,
        stride=1,
    ):
        super().__init__()
        self._bn_mom = 0.1
        self._bn_eps = 0.01
        self._se_ratio = 0.25
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride

        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        final_oup = self._output_filters

        # Expansion phase
        if self._expand_ratio != 1:
            self._expand_conv = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        padding = (ksize - 1) // 2
        self._depthwise_conv = nn.Conv2d(
            oup,
            oup,
            kernel_size=self._kernel_size,
            stride=stride,
            groups=oup,
            bias=False,
            padding=padding,
        )
        self._bn1 = nn.BatchNorm2d(oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer
        num_squeezed_channels = max(1, int(inp * self._se_ratio))
        self._se_reduce = nn.Conv2d(oup, num_squeezed_channels, kernel_size=1)
        self._se_expand = nn.Conv2d(num_squeezed_channels, oup, kernel_size=1)

        # Output phase
        self._project_conv = nn.Conv2d(oup, final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(final_oup, momentum=self._bn_mom, eps=self._bn_eps)

        self._swish = nn.SiLU()

    def forward(self, inputs):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._swish(self._se_reduce(x_squeezed))
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection
        if self._stride == 1 and self._input_filters == self._output_filters:
            x = x + inputs
        return x
