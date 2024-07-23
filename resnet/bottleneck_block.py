import torch
from torch import nn


class BottleneckBlock(nn.Module):
    """
    BasicBlock of Resnet. Consists of 2 convolutions
    """

    def __init__(self, in_features, out_features, downsample: bool = False):
        super().__init__()
        self.in_features = in_features
        self.intermediate_features = out_features // 4
        self.out_features = out_features

        self.downsample = downsample

        self.third_conv_stride = 1
        if self.downsample:
            self.third_conv_stride = 2

        self.conv1 = nn.Conv2d(
            self.in_features,
            self.intermediate_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.intermediate_features)
        self.conv2 = nn.Conv2d(
            self.intermediate_features,
            self.intermediate_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.intermediate_features)
        self.conv3 = nn.Conv2d(
            self.intermediate_features,
            self.out_features,
            kernel_size=1,
            stride=self.third_conv_stride,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.out_features)

        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    self.in_features,
                    self.out_features,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_features),
            )
        elif self.in_features != self.out_features:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(
                    self.in_features,
                    self.out_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_features),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block
        """
        batch_size = x.size(0)
        image_dim = x.size(2)

        shortcut = x
        assert shortcut.shape == (batch_size, self.in_features, image_dim, image_dim)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        assert x.shape == (batch_size, self.intermediate_features, image_dim, image_dim)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        assert x.shape == (batch_size, self.intermediate_features, image_dim, image_dim)

        x = self.conv3(x)
        x = self.bn3(x)
        # assert x.shape == (batch_size, self.out_features, image_dim, image_dim)

        if self.downsample or self.in_features != self.out_features:
            shortcut = self.downsample_layer(shortcut)
            # assert shortcut.shape == (
            #     batch_size,
            #     self.out_features,
            #     image_dim // 2,
            #     image_dim // 2,
            # )
        else:
            shortcut = x
            assert shortcut.shape == (
                batch_size,
                self.out_features,
                image_dim,
                image_dim,
            )

        x += shortcut
        x = self.relu(x)
        # assert x.shape == (batch_size, self.out_features, image_dim, image_dim)

        return x
