import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from vgg.vgg import VGG19

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE}")


def run(
    model: nn.Module,
    num_epochs: int,
    batch_size: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    percent_total: float = 1.0,
) -> None:
    """
    Trains, validates, and tests the model
    """

    train_loader, val_loader = get_train_val_dataloader(
        batch_size, percent_total=percent_total
    )

    for i in range(num_epochs):
        print(f"Training epoch {i + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer)
        print(f"Validating epoch {i + 1}/{num_epochs}")
        validate(model, val_loader, criterion)

    test_loader = get_test_dataloader(batch_size)
    print("Testing")
    test(model, test_loader, criterion)


def get_train_val_dataloader(
    batch_size: int, percent_train: float = 0.8, percent_total: float = 1.0
) -> tuple[DataLoader, DataLoader]:
    """
    Returns the training and validation dataloaders
    """
    cifar10 = CIFAR10(
        root="data/cifar10/training",
        train=True,
        download=True,
        transform=transform_data(),
    )

    # Split the training set into training and validation
    train_size = int(percent_train * len(cifar10) * percent_total)
    val_size = int(len(cifar10) * percent_total - train_size)
    print(
        f"Training on {train_size} samples, validating on {val_size} samples"
        f" out of {len(cifar10)} samples"
    )
    train_cifar10, val_cifar10 = random_split(cifar10, [train_size, val_size])
    # train_cifar10, val_cifar10 = train_test_split(
    #     cifar10, train_size=train_size, test_size=val_size
    # ) # TODO: this is not working, need to fix for testing

    train_loader = DataLoader(train_cifar10, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_cifar10, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_dataloader(batch_size: int) -> DataLoader:
    """
    Returns the test dataloader
    """
    test_cifar10 = CIFAR10(
        root="data/cifar10/test", train=False, download=True, transform=transform_data()
    )
    test_loader = DataLoader(test_cifar10, batch_size=batch_size, shuffle=False)

    return test_loader


def transform_data() -> transforms.Compose:
    """
    Returns the transformations to apply to the data
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            # transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Trains the model for an epoch
    """
    model.train(True)

    train_loss, num_correct, accuracy, num_total = 0, 0, 0, 0
    with tqdm(train_loader, total=len(train_loader)) as tepoch:
        for batch_idx, (images, labels) in enumerate(tepoch):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

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
            num_correct += (predicted == labels).sum().item()
            num_total += labels.size(0)
            accuracy = num_correct / num_total
            tepoch.set_postfix(loss=train_loss / (batch_idx + 1), accuracy=accuracy)


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> None:
    """
    Validates the model
    """
    model.eval()
    val_loss, num_correct, accuracy, num_total = 0, 0, 0, 0
    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader)) as vepoch:
            for batch_idx, (images, labels) in enumerate(vepoch):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                # prints number of each label
                for i in range(10):
                    print(f"Number of label {i}: {torch.sum(labels == i)}")

                # forward
                outputs = model(images)
                loss = criterion(outputs, labels)

                # loss
                val_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(outputs, 1)
                num_correct += (predicted == labels).sum().item()
                num_total += labels.size(0)
                accuracy = num_correct / num_total
                vepoch.set_postfix(loss=val_loss / (batch_idx + 1), accuracy=accuracy)


def test(model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> None:
    """
    Tests the model on the test set
    """
    model.eval()
    test_loss, num_correct, accuracy, num_total = 0, 0, 0, 0
    with torch.no_grad():
        with tqdm(test_loader, total=len(test_loader)) as tepoch:
            for batch_idx, (images, labels) in enumerate(tepoch):
                tepoch.set_description("Testing")

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                output = model(images)
                loss = criterion(output, labels)

                # Accuracy
                _, predicted = torch.max(output, 1)
                num_correct += (predicted == labels).sum().item()
                num_total += labels.size(0)
                accuracy = num_correct / num_total

                test_loss += loss.item()
                tepoch.set_postfix(loss=test_loss / (batch_idx + 1), accuracy=accuracy)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss/len(test_loader)}")
    print("Finished testing!")


if __name__ == "__main__":
    # Hyperparameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 128
    MOMENTUM = 0.9
    LR = 1e-2
    WEIGHT_DECAY = 5e-3
    PERCENT_TOTAL = 1.0  # Use 100% of the dataset
    # PERCENT_TOTAL = 0.001  # Use .1% of the dataset

    # Model, loss, and optimizer
    NUM_CLASSES = 10
    vgg = VGG19(NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        vgg.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
    )
    run(
        vgg,
        NUM_EPOCHS,
        BATCH_SIZE,
        criterion,
        optimizer,
        percent_total=PERCENT_TOTAL,
    )
