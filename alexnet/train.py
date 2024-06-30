from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

# from torchvision.transforms import ToTensor, Normalize
from tqdm import tqdm

from alexnet.alexnet import AlexNet


def get_train_and_val_loader(
    batch_size: int, percent_train: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # TODO: Add data augmentations
    # turn PIL images into tensors and normalize them
    train_transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    cifar10 = CIFAR10(
        root="data/cifar10/training",
        train=True,
        download=True,
        transform=train_transform,
    )

    # Split the training set into training and validation
    train_size = int(percent_train * len(cifar10))
    val_size = len(cifar10) - train_size
    train_cifar10, val_cifar10 = random_split(cifar10, [train_size, val_size])

    train_loader = DataLoader(train_cifar10, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_cifar10, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_loader(batch_size: int) -> DataLoader:
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_cifar10 = CIFAR10(
        root="data/cifar10/test", train=False, download=True, transform=train_transform
    )
    test_loader = DataLoader(test_cifar10, batch_size=batch_size, shuffle=False)

    return test_loader


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    num_epochs = 25
    learning_rate = 0.01
    weight_decay = 0.0005
    momentum = 0.9

    # Model
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    # learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=3, gamma=0.1
    # )

    train_loader, val_loader = get_train_and_val_loader(batch_size=batch_size)
    test_loader = get_test_loader(batch_size=batch_size)
    # Training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        accuracy = 0
        total = 0
        with tqdm(train_loader, total=len(train_loader)) as tepoch:
            for batch_idx, (images, labels) in enumerate(tepoch):
                tepoch.set_description(f"Training epoch {epoch + 1}/{num_epochs}")
                images, labels = images.to(device), labels.to(device)

                # forward
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                # backward
                loss.backward()
                optimizer.step()

                # loss
                train_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                accuracy = correct / total
                tepoch.set_postfix(loss=train_loss / (batch_idx + 1), accuracy=accuracy)

        model.eval()
        val_loss = 0
        correct = 0
        accuracy = 0
        total = 0
        with torch.no_grad():
            with tqdm(val_loader, total=len(val_loader)) as vepoch:
                for batch_idx, (images, labels) in enumerate(vepoch):
                    vepoch.set_description("Validating")
                    images, labels = images.to(device), labels.to(device)

                    # forward
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # loss
                    val_loss += loss.item()
                    vepoch.set_postfix(loss=val_loss / (batch_idx + 1))

                    # Accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    accuracy = correct / total
                    vepoch.set_postfix(
                        loss=val_loss / (batch_idx + 1), accuracy=accuracy
                    )

    print("Finished training!")

    # Testing
    model.eval()
    test_loss = 0
    correct = 0
    accuracy = 0
    with torch.no_grad():
        with tqdm(test_loader, total=len(test_loader)) as tepoch:
            for batch_idx, (images, labels) in enumerate(tepoch):
                tepoch.set_description("Testing")

                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)

                # Accuracy
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()
                accuracy = correct / len(test_loader.dataset)

                test_loss += loss.item()
                tepoch.set_postfix(test_loss=test_loss / (batch_idx + 1))

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss/len(test_loader)}")
    print("Finished testing!")
