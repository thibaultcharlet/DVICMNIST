from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

import torchvision.transforms as T


MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
MNIST.resources = [
   ("/".join([MIRROR, url.split("/")[-1]]), md5)
   for url, md5 in MNIST.resources
]


@dataclass
class MNISTDataset:
    train = MNIST(
        root="data",
        train=True,
        transform=T.Compose([T.RandomRotation(25), T.ToTensor()]),
        download=True,
    )
    test = MNIST(
        root="data",
        train=False,
        transform=T.ToTensor(),
        download=True,
    )


@dataclass
class MNISTLoader:
    train = DataLoader(
        MNISTDataset.train,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test = DataLoader(
        MNISTDataset.test,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )