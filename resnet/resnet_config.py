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
    second_conv_count: int
    third_conv_count: int
    fourth_conv_count: int
    five_conv_count: int

    def __post_init__(self):
        assert self.block_type in ["basic", "bottleneck"]

    @staticmethod
    def from_resnet_size(resnet_size: int) -> "ResNetConfig":
        """
        Get ResNetConfig from resnet size
        """
        return resnet_size_to_config_dict[resnet_size]


resnet_size_to_config_dict = {
    18: ResNetConfig("basic", 2, 2, 2, 2),
    35: ResNetConfig("basic", 3, 4, 6, 3),
    50: ResNetConfig("bottleneck", 3, 4, 6, 3),
    101: ResNetConfig("bottleneck", 3, 4, 23, 3),
    152: ResNetConfig("bottleneck", 3, 8, 36, 3),
}
