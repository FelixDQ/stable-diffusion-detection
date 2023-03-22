import torch
import numpy as np
from torchvision import transforms
from src.dataloader import load_dataset
from src.experiment import sdd_path, REAL_LOC
from src.util import get_accuracy, get_device, JPEGcompression
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from tqdm import tqdm

CH_EPSILONS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]

TRANSFORMS = {
    "pad3": transforms.Pad(padding=3),
    "pad10": transforms.Pad(padding=10),
    "pad30": transforms.Pad(padding=30),
    "pad50": transforms.Pad(padding=50),
    "compress": transforms.Lambda(JPEGcompression),
    "grayscale": transforms.Grayscale(3),
    "colorjitter": transforms.ColorJitter(
        brightness=0.1, hue=0.01, saturation=0.1, contrast=0.1
    ),
    "randomaffine": transforms.RandomAffine(degrees=(0, 10), translate=(0, 0.15)),
}


def test_robustness(model_func, model_name: str, size: int, sdd_version: str):
    name = f"{model_name}_{sdd_version}"

    print("TESTING ROBUSTNESS FOR: ", name)
    device = get_device()

    print("LOADING MODEL")
    model, optimizer, epochs, learning_rate = model_func()
    model.load_state_dict(torch.load(f"./{name}.pt"))
    model.to(device)
    model = model.eval()

    evaluation_results = {}
    print("TESTING PERFORMANCE ON OTHER SD VERSIONS")
    for sdv in sdd_path.keys():
        print("TESTING ON SD VERSION: ", sdv)
        train_loader, test_loader = load_dataset(
            real_path=REAL_LOC,
            fake_path=sdd_path[sdv],
            batch_size=64,
            samples=50000,
            size=size,
        )
        print("DATASET LOADED")

        test_acc = 0.0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.reshape((labels.shape[0])).to(device)
            output = torch.softmax(model(images), dim=1)
            test_acc += get_accuracy(output, labels, labels.shape[0])
        test_accuracy = test_acc / len(test_loader)
        print(f"TEST ACCURACY ON {sdv}: ", test_accuracy)
        evaluation_results[sdv] = test_accuracy

    print()
    print("TESTING PERFORMANCE ON TRANSFORMS")
    for t in TRANSFORMS.keys():
        print("TESTING ON: ", t)
        train_loader, test_loader = load_dataset(
            real_path=REAL_LOC,
            fake_path=sdd_path[sdd_version],
            batch_size=32,
            samples=50000,
            size=size,
            extra_transforms=TRANSFORMS[t],
        )
        print("DATASET LOADED")
        test_acc = 0.0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.reshape((labels.shape[0])).to(device)
            output = torch.softmax(model(images), dim=1)
            test_acc += get_accuracy(output, labels, labels.shape[0])
        test_accuracy = test_acc / len(test_loader)
        print(f"TEST ACCURACY ON {t}: ", test_accuracy)
        evaluation_results[t] = test_accuracy

    print()
    print("TESTING PERFORMANCE ON CLEVER HANS ATTACKS")
    for eps in CH_EPSILONS:
        print("TESTING ON EPSILON: ", eps)
        train_loader, test_loader = load_dataset(
            real_path=REAL_LOC,
            fake_path=sdd_path[sdd_version],
            batch_size=64,
            samples=50000,
            size=size,
        )
        print("DATASET LOADED")

        test_acc = 0.0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            x_pgm = projected_gradient_descent(
                model, images, eps, (2.5 * eps) / 40, 40, np.inf
            )
            # CH attack
            labels = labels.reshape((labels.shape[0])).to(device)
            output = torch.softmax(model(x_pgm), dim=1)
            test_acc += get_accuracy(output, labels, labels.shape[0])
        test_accuracy = test_acc / len(test_loader)
        print(f"TEST ACCURACY ON {eps}: ", test_accuracy)
        evaluation_results[eps] = test_accuracy
