import torch
import torchvision
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

def resnet18Training(mentee_model, teacher, dataset, lr, epochs, client_name, global_epoch):
    if os.path.isdir(f"./runs/global_epoch{global_epoch}/{client_name}") == False:
        os.makedirs(f"./runs/global_epoch{global_epoch}/{client_name}")
    writer = SummaryWriter(log_dir=f"./runs/global_epoch{global_epoch}/{client_name}")
    # resnet18 = models.resnet18(pretrained=True)
    
    teacher.train()
    mentee_model.train()

    for param in teacher.parameters():
        param.requires_grad = False

    num_ftrs = teacher.fc.in_features
    teacher.fc = nn.Linear(num_ftrs, 10)
    
    criterion_mentor = nn.CrossEntropyLoss()
    optimizer_mentor = optim.SGD(teacher.parameters(), lr= lr, momentum=0.9)

    # Define loss function and optimizer for mentee model
    criterion_mentee = nn.CrossEntropyLoss()
    optimizer_mentee = optim.SGD(mentee_model.parameters(), lr=lr, momentum=0.9)
    
# Training loop

    for epoch in range(epochs):
        running_loss_mentor = 0.0
        running_loss_mentee = 0.0
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            optimizer_mentor.zero_grad()
            optimizer_mentee.zero_grad()

            # Forward pass for mentor
            outputs_mentor = teacher(inputs)
            loss_mentor = criterion_mentor(outputs_mentor, labels)

            # Forward pass for mentee
            outputs_mentee = mentee_model(inputs)
            loss_mentee = criterion_mentee(outputs_mentee, labels)

            # Knowledge distillation loss
            temperature = 5
            soft_outputs_mentor = nn.functional.softmax(outputs_mentor / temperature, dim=1)
            soft_outputs_mentee = nn.functional.softmax(outputs_mentee / temperature, dim=1)
            distillation_loss = nn.functional.kl_div(soft_outputs_mentor.log(), soft_outputs_mentee, reduction='batchmean')
            total_loss = loss_mentee + distillation_loss

            # Backward pass for mentor
            total_loss.backward(retain_graph=True)  # Retain graph for backward pass of mentee

            # Backward pass for mentee
            loss_mentee.backward()

            # Optimizer step for mentor
            optimizer_mentor.step()

            # Optimizer step for mentee
            optimizer_mentee.step()

            # Log loss to TensorBoard

            running_loss_mentor += loss_mentor.item()
            running_loss_mentee += loss_mentee.item()

            # writer.add_scalar('mentor_training_loss', loss_mentor.item(), epoch*len(dataset)+1)
            # writer.add_scalar(f"{client_name} loss", loss_mentee.item(), epoch*len(dataset)+1)
    
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] mentor loss: %.3f, mentee loss: %.3f' %
                    (epoch + 1, i + 1, running_loss_mentor / 100, running_loss_mentee / 100))
                # writer.add_scalar('mentor_training_loss', loss_mentor.item(), epoch * len(train_loader) + i)
                writer.add_scalar(f'{client_name} loss', loss_mentee.item(), epoch * len(dataset) + i)
                running_loss_mentor = 0.0
                running_loss_mentee = 0.0
                break
        print(f"completed epoch:{epoch} ")
    print('Finished Training')
    
    return mentee_model.state_dict(), teacher.state_dict()

    
    # torch.save(mentee_model.state_dict(), f"models/{client_name}.pth")

    