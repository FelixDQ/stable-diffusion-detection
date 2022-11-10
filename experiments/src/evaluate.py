import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_model(model, image_loader, device):
    model.eval()
    y_pred = []
    y_true = []

    for i, (images, labels) in enumerate(image_loader):
        images = images.to(device)
        labels = torch.zeros((labels.shape[0], 1)).to(device)
        outputs = torch.sigmoid(model(images))
        y_pred.append(torch.round(outputs).view(-1).cpu().detach().numpy())
        y_true.append(labels.view(-1).cpu().detach().numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    cm = confusion_matrix(y_true, y_pred)
    return cm