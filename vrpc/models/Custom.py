import torch.nn.functional as F
import torch.nn as nn
import torch

from .common.convolution import BasicConvBlock, CustomConvBlock


class BasicStage(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.avgpool = nn.AvgPool2d(2, 2)
        self.shrink = nn.AdaptiveAvgPool2d(1)

        # drop resolution faster to replace last avg_pool

        # used Conv layer as Linear

        # more research on layer depth

        # make large resolution faster

        self.conv1 = BasicConvBlock(in_channels=3, out_channels=8, kernel=3)
        self.expand1 = nn.Conv2d(3, 8, 1)
        self.linear1 = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 32, bias=False),
            nn.GELU(),
        )

        self.conv2 = CustomConvBlock(in_channels=8, out_channels=32)
        self.expand2 = nn.Conv2d(3, 32, 1)
        self.linear2 = nn.Sequential(
            nn.LayerNorm(32),
            nn.Linear(32, 128, bias=False),
            nn.GELU(),
        )

        self.conv3 = CustomConvBlock(in_channels=32, out_channels=128)
        self.expand3 = nn.Conv2d(3, 128, 1)
        self.linear3 = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 512, bias=False),
            nn.GELU(),
        )

        self.conv4 = CustomConvBlock(in_channels=128, out_channels=512)
        self.expand4 = nn.Conv2d(3, 512, 1)
        self.linear4 = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512 * 3 * 3, bias=False),
            nn.GELU(),
        )

        # self.adapt_avgpool = nn.AdaptiveAvgPool2d(image_size // 16)
        # self.s_att = Encoder(
        #     seq_length=(image_size // 16) ** 2,
        #     num_layers=1,
        #     num_heads=4,
        #     hidden_dim=512,
        #     mlp_dim=64,
        # )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(512 * 3 * 3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.avgpool(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        l1 = self.shrink(x)
        l1 = self.linear1(torch.flatten(l1, 1))
        x += self.expand1(identity)

        identity = self.avgpool(identity)
        x = self.conv2(x)
        x = self.avgpool(x)
        l2 = self.shrink(x)
        l2 = self.linear2(torch.flatten(l2, 1) + l1)
        x += self.expand2(identity)

        identity = self.avgpool(identity)
        x = self.conv3(x)
        x = self.avgpool(x)
        l3 = self.shrink(x)
        l3 = self.linear3(torch.flatten(l3, 1) + l2)
        x += self.expand3(identity)

        identity = self.avgpool(identity)
        x = self.conv4(x)
        x = self.avgpool(x)
        l4 = self.shrink(x)
        l4 = self.linear4(torch.flatten(l4, 1) + l3)
        x += self.expand4(identity)

        # y = self.adapt_avgpool(x)
        # y = torch.flatten(y, 2).transpose(1, 2)
        # y = self.s_att(y).transpose(1, 2)
        # y = F.adaptive_avg_pool1d(y, 3)
        # y = torch.flatten(y, 1)

        x = F.adaptive_avg_pool2d(x, 3)
        x = torch.flatten(x, 1)

        x = self.classifier(x + l4)

        return x
