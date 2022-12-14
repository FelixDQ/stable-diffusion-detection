from src.util import get_accuracy

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.modules.loss._Loss,
    device: torch.device,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    inception: bool = False,
):
    for epoch in range(epochs):
        model.train()
        train_running_loss = 0.0
        train_acc = 0.0

        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(torch.float32).to(device)

            ## forward + backprop + loss
            if inception:
                output, aux_output = model(images)

                output = torch.sigmoid(output)
                aux_output = torch.sigmoid(aux_output)

                loss1 = criterion(output, labels)
                loss2 = criterion(aux_output, labels)
                loss = loss1 + 0.4*loss2

            else:
                output = model(images)
                output = torch.sigmoid(output)
                loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(output, labels, labels.shape[0])

        print('Epoch: {} | Training Loss: {:.6f} | Training Accuracy: {:.6f}'.format(
            epoch + 1,
            train_running_loss / len(train_loader),
            train_acc / len(train_loader),
        ))

    # Evaluate on test set
    model = model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = torch.sigmoid(model(images))
        test_acc += get_accuracy(output, labels, labels.shape[0])

    training_accuacy = train_acc / len(train_loader)
    test_accuracy = test_acc / len(test_loader)

    return training_accuacy, test_accuracy
