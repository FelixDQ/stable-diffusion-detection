from torchvision import models
import timm
from torch import nn
import torch

def get_inception_model():
    model = models.inception_v3()
    model.AuxLogits.fc = nn.Linear(768, 2)
    model.fc = nn.Linear(2048, 2)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    return model, optimizer, 30

def get_xception_model():
    model = timm.create_model('xception', pretrained=False, num_classes=2)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    return model, optimizer, 30

def get_densenet_model():
    model = timm.create_model('densenet121', pretrained=False, num_classes=2)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    return model, optimizer, 30

def get_vit_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    return model, optimizer, 30