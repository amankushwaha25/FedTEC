'''
    The script contains the implementation of transfer learning using some pre trained models.
    Current resnet18 and resnet50 are used
'''
import torch
import torchvision
from model import model
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

