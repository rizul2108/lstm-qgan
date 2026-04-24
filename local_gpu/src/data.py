import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

IMAGE_SIZE = 784
IMAGE_H    = 28
IMAGE_W    = 28
PATCH_SIZE = 196
N_PATCHES  = IMAGE_SIZE // PATCH_SIZE


def get_mnist_dataloader(batch_size=128, data_dir="./data", digit=None, subset_size=None, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    if digit is not None:
        dataset = Subset(dataset, [i for i, (_, y) in enumerate(dataset) if y == digit])
    if subset_size is not None:
        dataset = Subset(dataset, list(range(min(subset_size, len(dataset)))))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)


def flat_to_image(flat, image_h=IMAGE_H, image_w=IMAGE_W):
    return flat.view(flat.shape[0], 1, image_h, image_w)


def denormalise(images):
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)


def get_real_batch(dataloader, device):
    images, _ = next(iter(dataloader))
    return images.to(device).view(images.size(0), -1)
