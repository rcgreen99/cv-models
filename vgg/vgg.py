import torch

from torch import nn

# from torchvision.models import VGG


class VGG19(nn.Module):
    """
    VGG19 model implementation
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes),
            # nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        assert x.shape == (batch_size, 3, 224, 224)

        # First block
        x = self.conv1(x)
        assert x.shape == (batch_size, 64, 224, 224)
        x = self.maxpool(x)
        assert x.shape == (batch_size, 64, 112, 112)

        # Second block
        x = self.conv2(x)
        assert x.shape == (batch_size, 128, 112, 112)
        x = self.maxpool(x)
        assert x.shape == (batch_size, 128, 56, 56)

        # Third block
        x = self.conv3(x)
        assert x.shape == (batch_size, 256, 56, 56)
        x = self.maxpool(x)
        assert x.shape == (batch_size, 256, 28, 28)

        # Fourth block
        x = self.conv4(x)
        assert x.shape == (batch_size, 512, 28, 28)
        x = self.maxpool(x)
        assert x.shape == (batch_size, 512, 14, 14)

        # Fifth block
        x = self.conv5(x)
        assert x.shape == (batch_size, 512, 14, 14)
        x = self.maxpool(x)
        assert x.shape == (batch_size, 512, 7, 7)

        x = nn.AdaptiveAvgPool2d((7, 7))(x)

        x = torch.flatten(x, 1)

        # Classifier
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    BATCH_SIZE = 16
    NUM_CLASSES = 10
    model = VGG19(num_classes=NUM_CLASSES)
    model.forward(torch.rand(BATCH_SIZE, 3, 224, 224))
    print("Model forward pass successful!")
