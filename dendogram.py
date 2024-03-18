import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import json
import os
# Your data
def generate_cluster(data, round, threshold = 0.015, save_image = False, save_info = False):
    """
        The function to create clusters based on the thresold values.
        Parameter:
        data: list of tuples, each tuple is of shape (client_name1, client_name2, similarity score)
        thresold: float

        Return: 
        A dictionary having cluster names as key and values are the client names.
    """
    data = [
        ('client_0', 'client_1', 0.002697513),
        ('client_0', 'client_2', -0.0002878192),
        ('client_0', 'client_3', 0.001900201),
        ('client_0', 'client_4', -0.008431232),
        ('client_1', 'client_2', -0.00040878973),
        ('client_1', 'client_3', 0.0021052486),
        ('client_1', 'client_4', -0.0025824648),
        ('client_2', 'client_3', 0.0041731466),
        ('client_2', 'client_4', -0.010167966),
        ('client_3', 'client_4', -0.0029476644)
    ]

    # Extracting unique clients
    clients = sorted(set([item[0] for item in data] + [item[1] for item in data]))

    # Creating a distance matrix
    n = len(clients)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                for item in data:
                    if (item[0] == clients[i] and item[1] == clients[j]) or (item[0] == clients[j] and item[1] == clients[i]):
                        distances[i, j] = item[2]
                        break

    # Hierarchical clustering
    linkage_matrix = linkage(distances, method='complete')

    # Plotting dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=clients)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Clients')
    plt.ylabel('Distance')
    # plt.show()

    # Cutting dendrogram at a distance threshold
    # threshold = 0.015 # Adjust this threshold as needed
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
  
    # Printing final clusters
    cluster_dict = {}
    
    for client, cluster_id in zip(clients, clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(client)

    if save_image:
        if not os.path.exists("./dendogram"):
            os.makedirs("./dendogram")
        plt.savefig(f"./dendogram/{round}.png")
    

    if save_info:
        if not os.path.exists("./clusterInformation"):
            os.makedirs("./clusterInformation")
        cluster_dict_str_keys = {str(k): v for k, v in cluster_dict.items()}

    # Saving final clusters to a JSON file
        with open(f"./clusterInformation/{round}.json", "w") as f:
            json.dump(cluster_dict_str_keys, f, indent=4)
    
    return cluster_dict
    # print("Final Clusters:")
    # for cluster_id, members in cluster_dict.items():
    #     print(f"Cluster {cluster_id}: {', '.join(members)}")


    

if __name__ == "__main__":
    print(generate_cluster(None, 1, save_image=True, save_info=True))