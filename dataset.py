import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Download CIFAR-10 dataset
def data_generator(num_clients=2):
    """
    The function downloads the CIFAR10 dataset and splits the 
    dataset into number of clients

    Parameter:
        num_client : (int)
        concentration param : (float) to be decided 
    returns: list of splitted dataset
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # Define number of clients and labels
    
    num_labels = len(trainset.classes)

    print(f"Number of labels: {num_labels}")

    # Generate distribution for each client
    dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(torch.ones(num_labels))
    client_label_distributions = dirichlet_distribution.sample(torch.Size([num_clients]))

    
    print(f"concentration parameters : {dirichlet_distribution.concentration}")
    print(f"tensror: {torch.ones(num_labels)}")
    print(f"client label distribution: {client_label_distributions}")

    # Split dataset based on client labels
    client_datasets = []
    for i in range(num_clients):
        label_counts = (client_label_distributions[i] * len(trainset)).int()
        indices = torch.multinomial(torch.ones(len(trainset)), label_counts.sum(), replacement=False)
        subset = Subset(trainset, indices)
        client_datasets.append(subset)

    # Create data loaders for each client dataset
    client_loaders = []
    for dataset in client_datasets:
        loader = DataLoader(dataset, shuffle=True, num_workers=2)
        client_loaders.append(loader)

    return client_loaders



# def heatmap_generator(client_label_distribution, ):
#     heatmap_matrix = client_label_distributions.numpy()

#     # Plot heatmap
#     plt.figure(figsize=(10, 6))
#     plt.imshow(heatmap_matrix, cmap='viridis', interpolation='nearest')

#     # Add labels
#     plt.title('Label Distribution for Each Client')
#     plt.xlabel('Class')
#     plt.ylabel('Client')
#     plt.colorbar(label='Probability')

#     # Add annotations
#     for i in range(num_clients):
#         for j in range(num_labels):
#             plt.text(j, i, f'{heatmap_matrix[i, j]:.2f}', ha='center', va='center', color='white')

#     plt.xticks(np.arange(num_labels), range(num_labels))
#     plt.yticks(np.arange(num_clients), range(num_clients))

#     plt.show()

if __name__ == "__main__":
    data_generator(2)