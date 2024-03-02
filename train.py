import torch
import torchvision
from model import model
from torch.utils.tensorboard import SummaryWriter
from transfer_learning import resnet18Training
from dataset import data_generator

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

    def localTransferLearning(self):
        resnet18Training(self.net, self.dataset,self.lr, self.epochs, self.name)
        

class server():
    '''
        The server class represents the server architecture in federated learning.
        Constructors:
            num_clients(int) : number of clients in the learning process
            lr (float) : the learning parameter of each client (same for every client)
    '''
    def __init__(self, num_clients, lr) -> None:
        self.net = model()
        self.num_clients = num_clients
        self.clients =  [client(self.net, lr, f"client_{i}") for i in range(self.num_clients)]


if __name__ == "__main__":
    
    num_clients = 2    #command line argument 
    lr = 0.01          #command line argument
    server = server(num_clients, lr)

    # generate the data for the clinet according to the dirichlet distribution
    client_data_list = data_generator(num_clients)

    for i,client in enumerate(server.clients):
        client.dataset = client_data_list[i]
    
    for client in server.clients:
        client.localTransferLearning()