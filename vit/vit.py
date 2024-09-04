import torch
from torch import nn

from torchvision.models import vit_b_16


class ViT(nn.Module):
    """
    Vision Transformer according the the original "An Image is Worth 16x16 Words" paper
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


if __name__ == "__main__":
    NUM_CHANNELS = 3
    BATCH_SIZE = 16
    NUM_CLASSES = 10
    model = ViT(NUM_CHANNELS, NUM_CLASSES)
    model.forward(torch.rand(BATCH_SIZE, 3, 224, 224))
    print("Model forward pass successful!")
