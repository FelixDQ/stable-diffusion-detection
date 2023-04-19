from torchvision import models
import timm
from torch import nn
import torch


def get_xception_model():
    model = timm.create_model('xception', pretrained=False, num_classes=2)
    LEARNING_RATE = 1e-4 / 10
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    return model, optimizer, 30, LEARNING_RATE

def get_vit_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    LEARNING_RATE = 1e-5 / 10
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, optimizer, 30, LEARNING_RATE

def get_convnext_model():
    model = timm.create_model('convnext_small', pretrained=False, num_classes=2)
    LEARNING_RATE = 5e-5 / 10
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    return model, optimizer, 30, LEARNING_RATE