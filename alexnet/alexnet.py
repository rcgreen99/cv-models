import torch
from torch import nn

# import cifar10


class AlexNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x) -> torch.Tensor:
        # features
        batch_size = x.size(0)

        x = self.conv1(x)
        assert x.shape == (batch_size, 96, 55, 55)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        assert x.shape == (batch_size, 96, 27, 27)

        x = self.conv2(x)
        assert x.shape == (batch_size, 256, 27, 27)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        assert x.shape == (batch_size, 256, 13, 13)

        x = self.conv3(x)
        assert x.shape == (batch_size, 384, 13, 13)
        x = nn.ReLU()(x)
        assert x.shape == (batch_size, 384, 13, 13)

        x = self.conv4(x)
        assert x.shape == (batch_size, 384, 13, 13)
        x = nn.ReLU()(x)
        assert x.shape == (batch_size, 384, 13, 13)

        x = self.conv5(x)
        assert x.shape == (batch_size, 256, 13, 13)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        assert x.shape == (batch_size, 256, 6, 6)

        # # avgpool
        x = x.reshape(x.size(0), -1)
        assert x.shape == (batch_size, 256 * 6 * 6)

        # classifier
        x = nn.Dropout(self.dropout)(x)
        x = self.fc1(x)
        assert x.shape == (batch_size, 4096)
        x = nn.ReLU()(x)

        x = nn.Dropout(self.dropout)(x)
        x = self.fc2(x)
        assert x.shape == (batch_size, 4096)
        x = nn.ReLU()(x)

        x = self.fc3(x)
        assert x.shape == (batch_size, self.num_classes)

        return x


if __name__ == "__main__":
    batch_size = 16
    num_classes = 10
    model = AlexNet(num_classes=num_classes)
    model.forward(torch.rand(batch_size, 3, 227, 227))
    print("Model forward pass successful!")
