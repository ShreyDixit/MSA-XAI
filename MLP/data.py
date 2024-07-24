import copy
import torchvision.transforms as T
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset


def get_test_train_ds():
    mnist_transforms = T.Lambda(lambda x: torch.flatten(x, 1) / 255)

    train_data = datasets.MNIST(
        root="data",
        train=True,
        transform=mnist_transforms,
        download=True,
    )

    test_data = datasets.MNIST(root="data", train=False, transform=mnist_transforms)

    return mnist_transforms, train_data, test_data


def get_tensor_ds(device):
    mnist_transforms, train_data, test_data = get_test_train_ds()

    train_ds = TensorDataset(
        mnist_transforms(train_data.data.to(device)), train_data.targets.to(device)
    )
    test_ds = TensorDataset(
        mnist_transforms(test_data.data.to(device)), test_data.targets.to(device)
    )

    return train_ds, test_ds


def get_data_loaders(device):
    train_ds, test_ds = get_tensor_ds(device)

    trainloader = DataLoader(train_ds, batch_size=64, shuffle=True)

    testloader = DataLoader(test_ds, batch_size=1024)

    return trainloader, testloader


def get_dataloader_copy(dataloader):
    return copy.deepcopy(dataloader)
