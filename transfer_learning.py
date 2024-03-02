import torch
import torchvision
from torchvision import models
import torch.nn as nn
from torch.optim import optim
from torch.utils.tensorboard import SummaryWriter
import os
def resnet18Training(mentee_model, dataset, lr, epochs, client_name):
    resnet18 = models.resnet18(pretrained=True)

    for param in resnet18.parameters():
        param.requires_grad = False

    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)

    mentee_model.train()
    
    criterion_mentor = nn.CrossEntropyLoss()
    optimizer_mentor = optim.SDG(resnet18.parameters(), lr = lr, momentum=0.9)

    criterion_mentee = nn.CrossEntropyLoss()
    optimizer_mentee = optim.SDG(mentee_model.parameters(), lr=lr, momentum=0.9)

    writer = SummaryWriter()

    for epoch in range(epochs):
        running_loss_mentor = 0.0
        running_loss_mentee = 0.0

        for i, data in enumerate(dataset):
            input, label = data
            optimizer_mentor.zero_grad()
            optimizer_mentee.zero_grad()

            output_mentor = resnet18(input)
            loss_mentor = criterion_mentor(output_mentor, label)

            output_mentee = mentee_model(input)
            loss_mentee = criterion_mentee(output_mentee, label)

            # knowledge distillation loss
            temperature = 5
            soft_outputs_mentor = nn.functional.softmax(output_mentor/temperature, dim=1)
            distillation_loss = nn.functional.softmax(output_mentee/temperature, dim=1) 
            total_loss = loss_mentee + distillation_loss

            #backpropagation
            total_loss.backward(retain_graph=True)

            loss_mentee.backward()

            optimizer_mentee.step()
            optimizer_mentor.step()

            writer.add_scalar('mentor_training_loss', loss_mentor.item(), epoch*len(dataset)+1)
            writer.add_scalar('mentee_training_loss', loss_mentee.item(), epoch*len(dataset)+1)

            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] mentor loss: %.3f, mentee loss: %.3f' %
                    (epoch + 1, i + 1, running_loss_mentor / 100, running_loss_mentee / 100))
                running_loss_mentor = 0.0
                running_loss_mentee = 0.0
    print('Finished Training')

    
    torch.save(mentee_model.state_dict(), f"models/{client_name}.pth")

    