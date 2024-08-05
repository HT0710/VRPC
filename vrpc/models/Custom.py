import math

import torch.nn.functional as F
import torch.nn as nn
import torch

from .common.attention import ScaledDotProductAttention, ChannelAttention, Encoder
from .common.convolution import BasicConvBlock, DualConvBlock, BottleneckBlock


class BasicStage(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.conv1 = DualConvBlock(in_channels=3, out_channels=16, kernel=3)
        self.expand1 = nn.Conv2d(3, 16, 1)
        self.conv2 = DualConvBlock(in_channels=16, out_channels=32, kernel=3)
        self.expand2 = nn.Conv2d(3, 32, 1)
        self.conv3 = DualConvBlock(in_channels=32, out_channels=64, kernel=3)
        self.expand3 = nn.Conv2d(3, 64, 1)
        self.conv4 = DualConvBlock(in_channels=64, out_channels=128, kernel=3)
        self.expand4 = nn.Conv2d(3, 128, 1)
        self.c_att = ChannelAttention(in_channels=128, dropout=0)
        self.pos_embedding = nn.Parameter(torch.empty(1, 49, 128).normal_(std=0.02))
        self.s_att = Encoder(
            seq_length=49,
            num_layers=3,
            num_heads=2,
            hidden_dim=128,
            mlp_dim=256,
            attention_dropout=0,
            dropout=0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, 112)
        identity = x

        x = self.conv1(x)
        x = self.maxpool2d(x)
        identity = self.maxpool2d(identity)
        x += self.expand1(identity)

        x = self.conv2(x)
        x = self.maxpool2d(x)
        identity = self.maxpool2d(identity)
        x += self.expand2(identity)

        x = self.conv3(x)
        x = self.maxpool2d(x)
        identity = self.maxpool2d(identity)
        x += self.expand3(identity)

        x = self.conv4(x)
        x = self.maxpool2d(x)
        identity = self.maxpool2d(identity)
        x += self.expand4(identity)

        x = self.c_att(x)
        x = torch.flatten(x, 2).transpose(1, 2)
        x += self.pos_embedding
        x = self.s_att(x).transpose(1, 2)

        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class BasicStage0(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = image_size
        self.adapool = nn.AdaptiveAvgPool2d(111)
        self.maxpool2d = nn.MaxPool2d(2, stride=2)
        self.maxpool1d = nn.MaxPool1d(2, stride=2)
        self.conv1 = BasicConvBlock(in_channels=27, out_channels=32, kernel=1)
        self.c_att1 = ChannelAttention(in_channels=32, dropout=0)
        self.conv2 = BasicConvBlock(in_channels=32, out_channels=64, kernel=1)
        self.c_att2 = ChannelAttention(in_channels=64, dropout=0)
        self.conv3 = BasicConvBlock(in_channels=64, out_channels=128, kernel=1)
        self.c_att3 = ChannelAttention(in_channels=128, dropout=0)
        self.s_att = ScaledDotProductAttention(embed_dim=128, num_heads=8, dropout=0)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapool(x)
        B, C, H, W = x.shape
        num_patches = 9
        patch_size = int(math.sqrt(H * W / num_patches))
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(B, num_patches * C, patch_size, patch_size)
        x = self.conv1(x)
        x = self.c_att1(x)
        x = self.maxpool2d(x)
        x = self.conv2(x)
        x = self.c_att2(x)
        x = self.maxpool2d(x)
        x = self.conv3(x)
        x = self.c_att3(x)
        x = self.maxpool2d(x)
        x = x.reshape(*x.shape[:2], -1).transpose(1, 2)
        x = self.s_att(x, x, x).transpose(1, 2)
        x = self.fc(x)
        return x
