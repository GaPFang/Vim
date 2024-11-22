from torchvision import datasets, transforms

cifar100 = datasets.CIFAR100(
    root="./cifar100", train=True, download=True, transform=transforms.ToTensor()
)