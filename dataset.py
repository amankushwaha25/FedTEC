import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def data_generator(num_clients=10, alpha=0.1, datasetName = 'CIFAR10'):
    """
    The function downloads the CIFAR10 dataset and splits the 
    dataset into a number of clients based on the Dirichlet distribution.

    Parameters:
        num_clients: (int) Number of clients.
        alpha: (float) Concentration parameter for Dirichlet distribution.

    Returns:
        List of DataLoader objects for each client's dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if datasetName == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    elif datasetName == 'CIFAR100':   
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform)
    elif datasetName == 'TinyImageNet':
        trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
    
    num_labels = len(trainset.classes)

    # Generate distribution for each client using Dirichlet distribution
    dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(num_labels))
    client_label_distributions = dirichlet_distribution.sample(torch.Size([num_clients]))
    client_datasets = []
    for i in range(num_clients):
        label_counts = (client_label_distributions[i] * len(trainset)).int()
        indices = torch.multinomial(torch.ones(len(trainset)), label_counts.sum(), replacement=False)
        subset = Subset(trainset, indices)
        client_datasets.append(subset)

    # Create data loaders for each client dataset
    client_loaders = []
    for dataset in client_datasets:
        loader = DataLoader(dataset, shuffle=True, num_workers=2, batch_size=32)
        client_loaders.append(loader)
    return client_loaders

def generate_heatmap(client_loaders):
    """
    Generate a heatmap to visualize label quantities for each client.

    Parameters:
        client_loaders: List of DataLoader objects for each client's dataset.

    Returns:
        None (Displays the heatmap).
    """
    # Initialize an empty numpy array to store label quantities for each client
    num_clients = len(client_loaders)
    num_classes = 10  # CIFAR-10 has 10 classes
    label_quantities = np.zeros((num_clients, num_classes))

    # Calculate label quantities for each client
    for i, loader in enumerate(client_loaders):
        for _, labels in loader:
            for label in labels:
                label_quantities[i][label] += 1

    # Create a heatmap using Seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(label_quantities, annot=True, fmt='g', cmap='viridis')
    plt.xlabel('Classes')
    plt.ylabel('Clients')
    plt.title('Label Quantities for Each Client')
    plt.show()


if __name__ == "__main__":
   client_data = data_generator()
   print('done')
#    generate_heatmap(client_data)