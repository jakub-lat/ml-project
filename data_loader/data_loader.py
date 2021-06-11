import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_data(root):
    transform = transforms.Compose(
        transforms.ToTensor()
    )

    train_set = ImageFolder(
        root + "/Training",
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True
    )

    test_set = ImageFolder(
        root + "/Test",
        transform=transform,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=True
    )

    return (train_set, test_set), (train_loader, test_loader)
