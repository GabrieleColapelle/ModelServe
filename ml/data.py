import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import dgl.data

def load_mnist_data(root='data', flatten=True, batch_size=32):
    if flatten:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Lambda(lambda x: torch.flatten(x))]
        )
    else:
        transform = torchvision.transforms.ToTensor(),

    train_dataset = MNIST(root=root, download=True, transform=transform)
    test_dataset = MNIST(root=root, train=False,
                         download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader


def load_cora():
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.
    return data, dataset


def load_cora_dgl():
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    return g, dataset
