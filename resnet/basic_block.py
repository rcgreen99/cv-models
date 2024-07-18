import torch
from torch import nn


class BasicBlock(nn.Module):
    """
    BasicBlock of Resnet. Consists of 2 convolutions
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.downsample = self.check_if_downsample(in_features, out_features)

        if self.downsample:
            self.first_layer_stride = 2
        else:
            self.first_layer_stride = 1

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        image_dim = x.size(2)

        # Save input for residual connection
        if self.downsample:
            halved_image_dim = image_dim / 2
            first_layer_shape = (
                batch_size,
                self.out_features,
                halved_image_dim,
                halved_image_dim,
            )
            second_layer_shape = (
                batch_size,
                self.out_features,
                halved_image_dim,
                halved_image_dim,
            )
        else:
            first_layer_shape = (batch_size, self.out_features, image_dim, image_dim)
            second_layer_shape = first_layer_shape

        shortcut = x

        # First conv
        x = nn.Conv2d(
            self.in_features,
            self.out_features,
            kernel_size=3,
            stride=self.first_layer_stride,
            padding=1,
            bias=False,
        )(x)
        x = nn.BatchNorm2d(self.out_features)(x)
        x = nn.ReLU()(x)
        assert x.shape == first_layer_shape

        # Second conv
        x = nn.Conv2d(
            self.out_features,
            self.out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )(x)
        x = nn.BatchNorm2d(self.out_features)(x)
        x = nn.ReLU()(x)

        assert x.shape == second_layer_shape

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

        # residual
        x += shortcut
        x = nn.ReLU()(x)

        return x

    def check_if_downsample(self, in_features, out_features):
        if in_features == out_features:
            return False
        if in_features == out_features / 2:
            return True
        raise Exception(
            "Invalid values for in_features and out_features. Must be either the same or out_features half of in features"
        )
