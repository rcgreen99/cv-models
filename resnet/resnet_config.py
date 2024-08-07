from typing import List
from dataclasses import dataclass


@dataclass
class ResNetConfig:
    """
    Configuration for ResNet model

    Attributes:
        block_type: str, type of block to use
            available options: "basic", "bottleneck"
        second_conv_count: int, number of convolutions in second block
        third_conv_count: int, number of convolutions in third block
        fourth_conv_count: int, number of convolutions in fourth block
        five_conv_count: int, number of convolutions in fifth
    """

    block_type: str
    num_blocks_per_stage: List[int]
    num_final_features: int

    def __post_init__(self):
        assert self.block_type in ["basic", "bottleneck"]

    @staticmethod
    def from_resnet_size(resnet_size: int) -> "ResNetConfig":
        """
        Get ResNetConfig from resnet size
        """
        return resnet_size_to_config_dict[resnet_size]


# Dictionary to map resnet size to ResNetConfig
# 18 and 34 are use BasicBlocks ResNet models
# 50, 101, and 152 use BottleNeckBlocks ResNet models
resnet_size_to_config_dict = {
    18: ResNetConfig("basic", [2, 2, 2, 2], 512),
    34: ResNetConfig("basic", [3, 4, 6, 3], 512),
    50: ResNetConfig("bottleneck", [3, 4, 6, 3], 2048),
    101: ResNetConfig("bottleneck", [3, 4, 23, 3], 2048),
    152: ResNetConfig("bottleneck", [3, 8, 36, 3], 2048),
}
