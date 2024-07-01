import torch
from torch import nn


class VGG19(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 3, 224, 244)

        x = self.conv1(x)
        assert x.shape == (batch_size, 64, 75, 75)
        x = nn.MaxPool2d()

        x = nn.MaxPool2d((2, 2), stride=2)

        return x


if __name__ == "__main__":
    batch_size = 16
    num_classes = 10
    model = VGG19(num_classes=num_classes)
    model.forward(torch.rand(batch_size, 3, 224, 224))
    print("Model forward pass successful!")
