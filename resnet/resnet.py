from typing import List

import torch
from torch import nn

from resnet.basic_block import BasicBlock
from resnet.bottleneck_block import BottleneckBlock
from resnet.resnet_config import ResNetConfig


class ResNet(nn.Module):
    """
    Implementation of ResNet model
    """

    def __init__(self, in_channels: int, num_classes: int, resnet_depth: int) -> None:
        """
        Initialize ResNet model

        Args:
            num_classes: int, number of classes in the dataset
            resnet_depth: int, depth of resnet
                available options: 18, 34, 50, 101, 152
        """
        super().__init__()
        self.num_classes = num_classes
        self.resnet_config = ResNetConfig.from_resnet_size(resnet_depth)

        self.stem = self.get_stem(in_channels)
        self.features = self.get_features(self.resnet_config)
        self.classifier = self.get_classifier(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x: torch.Tensor, input tensor

        Returns:
            torch.Tensor, output tensor
        """
        batch_size = x.size(0)
        x = self.stem(x)
        assert x.shape == (batch_size, 64, 56, 56)
        x = self.features(x)
        assert x.shape == (batch_size, 512, 7, 7)
        x = self.classifier(x)
        assert x.shape == (batch_size, self.num_classes)
        return x

    def get_stem(self, in_channels: int) -> nn.Sequential:
        """
        Returns the stem layer for ResNet
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def get_features(self, resnet_config: ResNetConfig) -> nn.Sequential:
        """
        Based on the depth of resnet, return the feature extractor
        """
        # all ResNet's are 5 blocks deep
        # 18 and 34 have 2 layers in each block and use basic block
        # 50, 101, and 152 have 3 layers in each block and use bottleneck block
        # conv2
        if resnet_config.block_type == "basic":
            return self.build_basic_block_resnet(resnet_config)
        elif resnet_config.block_type == "bottleneck":
            raise NotImplementedError("BottleneckBlock not implemented yet")
            # return self.build_bottlenekc_block_resnet(resnet_config)
        else:
            raise ValueError("Invalid block type")

    def build_basic_block_resnet(self, resnet_config: ResNetConfig):
        features = nn.Sequential()
        in_features = 64
        for conv_layer_idx in range(4):  # 4 conv layers after stem
            num_blocks = resnet_config.num_blocks_per_stage[conv_layer_idx]
            out_features = 64 * 2**conv_layer_idx
            for block_idx in range(num_blocks):
                if block_idx != 0:
                    in_features = 64 * 2 ** (conv_layer_idx)
                features.add_module(
                    f"block{conv_layer_idx + 2}_{block_idx}",
                    BasicBlock(in_features, out_features),
                )
        return features

    def get_classifier(self, num_classes: int) -> nn.Sequential:
        """
        Return the classifier
        """
        # classifier
        return nn.Sequential(
            nn.AvgPool2d(7, 1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1),
        )


if __name__ == "__main__":
    BATCH_SIZE = 16
    NUM_CLASSES = 10
    RESNET_SIZE = 34
    model = ResNet(3, NUM_CLASSES, RESNET_SIZE)
    model.forward(torch.rand(BATCH_SIZE, 3, 224, 224))
    print("Model forward pass successful!")

    # return nn.Sequential(
    #     # Conv 2
    #     BasicBlock(64, 64),
    #     BasicBlock(64, 64),
    #     # Conv 3
    #     BasicBlock(64, 128),
    #     BasicBlock(128, 128),
    #     # Conv 4
    #     BasicBlock(128, 256),
    #     BasicBlock(256, 256),
    #     # Conv 5
    #     BasicBlock(256, 512),
    #     BasicBlock(512, 512),
    # )
