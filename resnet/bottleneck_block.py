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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block
        """
        batch_size = x.size(0)
        image_dim = x.size(2)

        shortcut = x
        assert shortcut.shape == (batch_size, self.in_features, image_dim, image_dim)

        # 1st conv
        x = nn.Conv2d(
            self.in_features,
            self.intermediate_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )(x)
        x = nn.BatchNorm2d(self.intermediate_features)(x)
        x = nn.ReLU()(x)
        assert x.shape == (
            batch_size,
            self.intermediate_features,
            image_dim,
            image_dim,
        )

        # 2nd conv
        x = nn.Conv2d(
            self.intermediate_features,
            self.intermediate_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )(x)
        x = nn.BatchNorm2d(self.intermediate_features)(x)
        x = nn.ReLU()(x)
        assert x.shape == (
            batch_size,
            self.intermediate_features,
            image_dim,
            image_dim,
        )

        # 3rd conv
        x = nn.Conv2d(
            self.intermediate_features,
            self.out_features,
            kernel_size=1,
            stride=self.third_conv_stride,
            padding=0,
            bias=False,
        )(x)
        x = nn.BatchNorm2d(self.out_features)(x)
        # assert x.shape == (batch_size, self.out_features, image_dim, image_dim)

        # if we downsampled, need to use projection shortcut
        if self.downsample:
            shortcut = nn.Conv2d(
                self.in_features,
                self.out_features,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=False,
            )(shortcut)
            shortcut = nn.BatchNorm2d(self.out_features)(shortcut)
            # assert shortcut.shape == (
            #     batch_size,
            #     self.out_features,
            #     half_image_dim,
            #     half_image_dim,
            # )
        # if this is the first block of the layer, we need to increase the number of features
        # to match the output of the block
        elif self.in_features != self.out_features:
            shortcut = nn.Conv2d(
                self.in_features,
                self.out_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )(shortcut)
            shortcut = nn.BatchNorm2d(self.out_features)(shortcut)
            # assert shortcut.shape == (
            #     batch_size,
            #     self.out_features,
            #     image_dim,
            #     image_dim,
            # )
        # residual
        x += shortcut
        x = nn.ReLU()(x)
        # assert x.shape == (batch_size, self.out_features, image_dim, image_dim)

        return x
