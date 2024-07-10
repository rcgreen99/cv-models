import torch
from torch import nn


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
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.resnet_depth = resnet_depth

        # self.stem = self.get_stem(in_channels)
        # self.features = self.get_features(resnet_depth)
        # self.classifier = self.get_classifier(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x: torch.Tensor, input tensor

        Returns:
            torch.Tensor, output tensor
        """
        # x = self.stem(x)
        # x = self.features(x)
        # x = self.classifier(x)

        batch_size = x.size(0)

        # stem
        x = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )(x)
        assert x.shape == (batch_size, 64, 112, 112)
        x = nn.BatchNorm2d(64)(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        assert x.shape == (batch_size, 64, 56, 56)

        # conv2
        x1 = x
        x = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)(x)
        assert x.shape == (batch_size, 64, 56, 56)
        x = nn.BatchNorm2d(64)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)(x)
        assert x.shape == (batch_size, 64, 56, 56)
        x = nn.BatchNorm2d(64)(x)
        x = nn.ReLU()(x)
        x += x1

        # conv3
        x2 = x
        x = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)(x)
        assert x.shape == (batch_size, 128, 28, 28)
        x = nn.BatchNorm2d(128)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)(x)
        assert x.shape == (batch_size, 128, 28, 28)
        x = nn.BatchNorm2d(128)(x)
        x = nn.ReLU()(x)

        # projection shortcut
        x2 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False)(x2)
        x2 = nn.BatchNorm2d(128)(x2)
        assert x2.shape == (batch_size, 128, 28, 28)
        x += x2

        # conv 4
        x3 = x
        x = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)(x)
        assert x.shape == (batch_size, 256, 14, 14)
        x = nn.BatchNorm2d(256)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)(x)
        assert x.shape == (batch_size, 256, 14, 14)
        x = nn.BatchNorm2d(256)(x)
        x = nn.ReLU()(x)

        # projection shorcut
        x3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False)(x3)
        x3 = nn.BatchNorm2d(256)(x3)
        assert x3.shape == (batch_size, 256, 14, 14)
        x += x3

        # conv 5
        x4 = x
        x = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)(x)
        assert x.shape == (batch_size, 512, 7, 7)
        x = nn.BatchNorm2d(512)(x)
        x = nn.ReLU()(x)
        x = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)(x)
        assert x.shape == (batch_size, 512, 7, 7)
        x = nn.BatchNorm2d(512)(x)
        x = nn.ReLU()(x)

        # projection shorcut
        x4 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False)(x4)
        x4 = nn.BatchNorm2d(512)(x4)
        assert x4.shape == (batch_size, 512, 7, 7)
        x += x4

        # classifier
        x = nn.AvgPool2d(7, 1)(x)
        x = x.view(x.size(0), -1)
        x = nn.Linear(512, self.num_classes)(x)
        x = nn.Softmax(dim=1)(x)
        assert x.shape == (batch_size, self.num_classes)

        return x

    def get_stem(self, in_channels: int = 3) -> nn.Sequential:
        """
        Returns the stem layer for ResNet
        """
        return nn.Sequential(
            # unknown stride and padding
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7),
            nn.BatchNorm2d(2, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def get_features(self, resnet_depth: int) -> nn.Sequential:
        """
        Based on the depth of resnet, return the feature extractor
        """
        pass

    def get_classifier(self, num_classes: int) -> nn.Sequential:
        """
        Return the classifier
        """
        return None


if __name__ == "__main__":
    batch_size = 16
    num_classes = 10
    resnet_size = 18
    model = ResNet(3, num_classes, resnet_size)
    model.forward(torch.rand(batch_size, 3, 224, 224))
    print("Model forward pass successful!")
