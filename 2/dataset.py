import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class CIFAR10PerPixelDataset(Dataset):
    def __init__(self, images, targets, augment=False):
        self.images = images
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.targets[idx]

        if self.augment:
            padded = F.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            img = padded[:, top : top + 32, left : left + 32]
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])

        return img, target


def get_cifar10_loaders(data_dir="./data", batch_size=128):
    train_raw = CIFAR10(root=data_dir, train=True, download=True)
    test_raw = CIFAR10(root=data_dir, train=False, download=True)

    train_images = np.transpose(train_raw.data, (0, 3, 1, 2)).astype(np.float32) / 255.0
    test_images = np.transpose(test_raw.data, (0, 3, 1, 2)).astype(np.float32) / 255.0

    pixel_mean = train_images.mean(axis=0, keepdims=True)
    train_images = torch.from_numpy(train_images - pixel_mean)
    test_images = torch.from_numpy(test_images - pixel_mean)

    train_targets = torch.tensor(train_raw.targets, dtype=torch.long)
    test_targets = torch.tensor(test_raw.targets, dtype=torch.long)

    train_dataset = CIFAR10PerPixelDataset(train_images, train_targets, augment=True)
    test_dataset = CIFAR10PerPixelDataset(test_images, test_targets, augment=False)

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    return train_loader, test_loader, pixel_mean
