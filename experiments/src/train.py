from src.util import get_accuracy
from src.dataloader import get_transforms
from tqdm import tqdm
import random

import torch
import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

def train_model(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.modules.loss._Loss,
    device: torch.device,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    learning_rate: float,
    size: int = 224,
    adv_training: bool = False,
):
    transforms, _ = get_transforms(size, already_tensor=True)
    def model_with_transforms(x):
        # model.eval()
        x_fgm = model(transforms(x))
        # model.train()
        return x_fgm

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=epochs)
    for epoch in range(epochs):
        model.train()
        train_running_loss = 0.0
        train_acc = 0.0

        for idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.reshape((labels.shape[0])).to(device)

            if torch.isnan(images).any():
                print("uh oh, nan in images. skipping batch...")
                continue

            if adv_training:
                if random.random() < 0.5:
                    images = fast_gradient_method(model_with_transforms, images, 0.1, np.inf, targeted=False)

                images = transforms(images)

            output = model(images)
            output = torch.softmax(output, dim=1)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()

            ## update model params
            optimizer.step()
            scheduler.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(output, labels, labels.shape[0])

        print('Epoch: {} | Training Loss: {:.6f} | Training Accuracy: {:.6f}'.format(
            epoch + 1,
            train_running_loss / len(train_loader),
            train_acc / len(train_loader),
        ))

    # Evaluate on test set
    # model = model.eval()
    # test_acc = 0.0
    # for i, (images, labels) in enumerate(test_loader):
    #     images = images.to(device)
    #     if adv_training:
    #           images = transforms(images)
    #     labels = labels.reshape((labels.shape[0])).to(device)
    #     output = torch.softmax(model(images), dim=1)
    #     test_acc += get_accuracy(output, labels, labels.shape[0])

    training_accuacy = train_acc / len(train_loader)
    # test_accuracy = test_acc / len(test_loader)
    test_accuracy = 0.0

    return training_accuacy, test_accuracy
