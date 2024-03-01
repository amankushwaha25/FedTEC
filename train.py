import torch
import torchvision
from model import model

class client():
    '''
        The client class represents client architecture in federated learning.
        Constructors:
            net (from model) : to initialise the client with a particular NN architecture
            lr (float) : Learning rate
            epochs (int) : local epochs for training the model locally 
    '''
    def __init__(self, net, lr, epochs) -> None:
        self.net = net
        self.epochs = epochs
        self.lr = lr

    def local_update(net):
        pass

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
        self.clients =  [client(self.net, lr) for _ in range(self.num_clients)]



