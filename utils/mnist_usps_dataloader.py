import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=32):
    common_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    # MNIST
    mnist_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=common_transform
    )


    # USPS
    usps_dataset = datasets.USPS(
        root='./data',
        train=True,
        download=True,
        transform=common_transform
    )

    # create dataloader
    source_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    target_loader = DataLoader(usps_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Done loader datasets!!!")

    return source_loader, target_loader