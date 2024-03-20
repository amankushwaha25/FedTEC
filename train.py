import torch
from torchvision import models
from model import model
from torch.utils.tensorboard import SummaryWriter
from transfer_learning import resnet18Training
from dataset import data_generator
import threading
from CKA import calculate_similarity
import os
import warnings
from itertools import combinations
from dendogram import generate_cluster
warnings.filterwarnings("ignore")
import time
import concurrent.futures
# %tensorboard --logdir=./runs1
class client():
    '''
        The client class represents client architecture in federated learning.
        Constructors:
            net (from model) : to initialise the client with a particular NN architecture
            lr (float) : Learning rate
            epochs (int) : local epochs for training the model locally 
    '''
    def __init__(self, net, lr, epochs, name) -> None:
        self.name = name
        self.net = net
        self.epochs = epochs
        self.lr = lr
        self.dataset = None
        self.teacher = models.resnet18(pretrained=True)

    def localTransferLearning(self, global_epoch):
        return resnet18Training(self.net, self.teacher, self.dataset,self.lr, self.epochs, self.name, global_epoch)
        
    def saveModel(self, global_epoch):
        if not os.path.exists(f"./models/global_epoch{global_epoch}"):
            os.makedirs(f"./models/global_epoch{global_epoch}")
        torch.save(self.net.state_dict(), f'./models/global_epoch{global_epoch}/{self.name}.pt')
class server():
    '''
        The server class represents the server architecture in federated learning.
        Constructors:
            num_clients(int) : number of clients in the learning process
            lr (float) : the learning parameter of each client (same for every client)
    '''
    def __init__(self, num_clients, lr, local_epochs) -> None:
        self.net = model()
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.clients =  [client(self.net, lr, self.local_epochs, f"client_{i}") for i in range(self.num_clients)]
        self.clusters = None
        self.record = {}

    def similarity_function(self, global_epoch):
       
        self.record = { client.name : client for client in self.clients}
        # print(self.record)
        time.sleep(10)
        datasets = data_generator(num_clients=500)
        path = f"./models/global_epoch{global_epoch}"

        all_models = []
     
        for file in os.listdir(path):
            if file.endswith('.pt'):
                model_instance = model()  # Instantiate the model class
                model_instance.load_state_dict(torch.load(os.path.join(path, file)))  # Load the model state dictionary
                all_models.append((file.split('.')[0], model_instance))
    
        combination = combinations(all_models, 2)
    
        pairwise_similarity = []
    
    # print("length of all_models=",len(all_models))
    # print("length of combination=",(combination))

        for pair in combination:
            pairwise_similarity.append(calculate_similarity(pair[0][0], pair[1][0], pair[0][1], pair[1][1], datasets[0]))

        self.clusters = generate_cluster(pairwise_similarity, round= global_epoch, save_image=True, save_info=True)
    
    def average(self):
        for _ , cluster_mem in self.clusters.items():
            average_model = {}
            # Initialize average_model with keys from the first member
            for key, value in self.record[cluster_mem[0]].net.state_dict().items():
                average_model[key] = torch.zeros_like(value)
    
            # Accumulate weighted sum of parameters
            for key in average_model:
                weighted_sum = sum(self.record[member].net.state_dict()[key] for member in cluster_mem)
                average_model[key] = weighted_sum / len(cluster_mem)
    
            for member in cluster_mem:
                self.record[member].net.load_state_dict(average_model)
                # print(self.record[member].net.state_dict())

        
        
# helper function for threads
def train_client(client, global_epoch):
    print(f"Training client {client.name}...")
    updated_client_model, updated_teacher_model = client.localTransferLearning(global_epoch)
    client.net.load_state_dict(updated_client_model)
    client.teacher.load_state_dict(updated_teacher_model)
    client.saveModel(global_epoch)
    print(f"Client {client.name} training completed.")

if __name__ == "__main__":
    
    num_clients = 3    #command line argument 
    lr = 0.01           #command line argument
    local_epochs =  2  #command line argument
    global_epoch = 0
    max_global_epoch = 2
    server = server(num_clients, lr, local_epochs)
   
    # generate the data for the clinet according to the dirichlet distribution
    client_data_list = data_generator(num_clients)

    for i,client in enumerate(server.clients):
        client.dataset = client_data_list[i]
    
    # actual training -> needs improvement
    while global_epoch != max_global_epoch:
        # for client in server.clients:
        #     updated_client_model, updated_teacher_model  = client.localTransferLearning(global_epoch)
        #     client.net.load_state_dict(updated_client_model)
        #     client.teacher.load_state_dict(updated_teacher_model)
        #     client.saveModel(global_epoch)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(train_client, client, global_epoch) for client in server.clients]
            # Wait for all futures to complete
            concurrent.futures.wait(futures)
        server.similarity_function(global_epoch)
        server.average()
        global_epoch += 1
        print(f"**************************************************completed global epoch {global_epoch} \
              **************************************************")
    
    print("**************************************************Completed federated learning**************************************************")
  
   