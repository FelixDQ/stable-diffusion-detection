from torchvision import models
from torch import nn

def get_inception_model():
    model = models.inception_v3()
    model.AuxLogits.fc = nn.Linear(768, 1)
    model.fc = nn.Linear(2048, 1)
    return model