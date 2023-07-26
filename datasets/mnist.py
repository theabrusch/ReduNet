import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .utils_data import filter_class
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def generate_mnist_concept_dataset(concept_classes: list[int], data_dir, train: bool, subset_size: int,
                                   random_seed: int) -> tuple:
    """
    Return a concept dataset with positive/negatives for MNIST
    Args:
        random_seed: random seed for reproducibility
        subset_size: size of the positive and negative subset
        concept_classes: the classes where the concept is present in MNIST
        data_dir: directory where MNIST is saved
        train: sample from the training set

    Returns:
        a concept dataset of the form X (features),y (concept labels)
    """
    dataset = datasets.MNIST(data_dir, train=train, download=True)
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset.transform = data_transform
    targets = dataset.targets
    mask = torch.zeros(len(targets))
    for idx, target in enumerate(targets):  # Scan the dataset for valid examples
        if target in concept_classes:
            mask[idx] = 1
    positive_idx = torch.nonzero(mask).flatten()
    negative_idx = torch.nonzero(1 - mask).flatten()
    positive_loader = torch.utils.data.DataLoader(dataset, batch_size=subset_size,
                                                  sampler=SubsetRandomSampler(positive_idx))
    negative_loader = torch.utils.data.DataLoader(dataset, batch_size=subset_size,
                                                  sampler=SubsetRandomSampler(negative_idx))
    positive_images, positive_labels = next(iter(positive_loader))
    negative_images, negative_labels = next(iter(negative_loader))
    X = np.concatenate((positive_images.cpu().numpy(), negative_images.cpu().numpy()), 0)
    y = np.concatenate((np.ones(subset_size), np.zeros(subset_size)), 0)
    np.random.seed(random_seed)
    rand_perm = np.random.permutation(len(X))
    return X[rand_perm], y[rand_perm]

def mnist2d_concepts(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    num_classes = 10
    return trainset, testset, num_classes

def mnistvector_concepts(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    num_classes = 10
    return trainset, testset, num_classes


def mnist2d_10class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    num_classes = 10
    return trainset, testset, num_classes

def mnist2d_5class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    trainset, num_classes = filter_class(trainset, [0, 1, 2, 3, 4])
    testset, _ = filter_class(testset, [0, 1, 2, 3, 4])
    num_classes = 5
    return trainset, testset, num_classes

def mnist2d_2class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    trainset, num_classes = filter_class(trainset, [0, 1])
    testset, _ = filter_class(testset, [0, 1])
    return trainset, testset, num_classes

def mnistvector_10class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    num_classes = 10
    return trainset, testset, num_classes

def mnistvector_5class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    trainset, num_classes = filter_class(trainset, [0, 1, 2, 3, 4])
    testset, _ = filter_class(testset, [0, 1, 2, 3, 4])
    return trainset, testset, num_classes

def mnistvector_2class(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten())
    ])
    trainset = datasets.MNIST(data_dir, train=True, transform=transform, download=True)
    testset = datasets.MNIST(data_dir, train=False, transform=transform, download=True)
    trainset, num_classes = filter_class(trainset, [0, 1])
    testset, _ = filter_class(testset, [0, 1])
    return trainset, testset, num_classes


if __name__ == '__main__':
    trainset, testset, num_classes = mnist2d_2class('./data/')
    trainloader  = DataLoader(trainset, batch_size=trainset.data.shape[0])
    print(trainset)
    print(testset)
    print(num_classes)

    batch_imgs, batch_lbls = next(iter(trainloader))
    print(batch_imgs.shape, batch_lbls.shape)
    print(batch_lbls.unique(return_counts=True))
