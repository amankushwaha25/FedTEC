from torch_cka import CKA
from model import model1, model2, model3
import torch
from dataset import data_generator 
import warnings
import os
from itertools import combinations 
import threading
import concurrent.futures
warnings.filterwarnings('ignore')

def calculate_similarity(client1_name, client2_name, model1, model2, dataset):
    
   
    cka = CKA(model1, model2, model1_name="model1",  model2_name="model2") 
    cka.compare(dataset)
    results = cka.export()
    
    print(f"{client1_name} and {client2_name} : ", results['CKA'].numpy().diagonal().sum())
    return (client1_name, client2_name, results['CKA'].numpy().diagonal().sum())

# if __name__ == "__main__":

    # datasets = data_generator(num_clients=500)
    # path = "./models"

    # all_models = []
    # similarity_table = []
    # for file in os.listdir(path):
    #     if file.endswith('.pt'):
    #         model_instance = model()  # Instantiate the model class
    #         model_instance.load_state_dict(torch.load(os.path.join(path, file)))  # Load the model state dictionary
    #         all_models.append((file.split('.')[0], model_instance))
    
    # combination = combinations(all_models, 2)
    
    # pairwise_similarity = []
    

    # for pair in combination:
    #     pairwise_similarity.append(calculate_similarity(pair[0][0], pair[1][0], pair[0][1], pair[1][1], datasets[0]))
    
    # print(pairwise_similarity)
