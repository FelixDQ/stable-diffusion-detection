from typing import Optional
import torch
import numpy as np
import os
from torchvision import transforms
from src.dataloader import load_dataset, get_transforms
from src.experiment import sdd_path, REAL_LOC
from src.util import get_accuracy, get_device, JPEGcompression, get_confusion_matrix
from autoattack import AutoAttack
from tqdm import tqdm
import json

LINF_EPSILONS = [4/255, 8/255]

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


def test_robustness(model_func, model_name: str, size: int, sdd_version: str, model_path: Optional[str] = None, model_suffix: Optional[str] = None, file_extension: str = "pt", extra_transforms=None):
    name = f"{model_name}_{sdd_version}"
    if model_suffix:
        name += f"_{model_suffix}"

    path = "."
    if model_path:
        path = model_path

    print("TESTING ROBUSTNESS FOR: ", name)
    device = get_device()

    print("LOADING MODEL")
    model, optimizer, epochs, learning_rate = model_func()
    model.load_state_dict(torch.load(os.path.join(path, f"./{name}.{file_extension}")))
    model.to(device)
    model = model.eval()

    evaluation_results = {}
    print("TESTING PERFORMANCE ON OTHER SD VERSIONS")
    for sdv in sdd_path.keys():
        print("TESTING ON SD VERSION: ", sdv)
        train_loader, test_loader = load_dataset(
            real_path=REAL_LOC,
            fake_path=sdd_path[sdv],
            batch_size=32,
            samples=50000,
            size=size,
            extra_transforms=extra_transforms,
        )
        print("DATASET LOADED")

        test_acc = 0.0
        cm = torch.zeros(2, 2, dtype=torch.int64).to(device)
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.reshape((labels.shape[0])).to(device)
            output = torch.softmax(model(images), dim=1)
            test_acc += get_accuracy(output, labels, labels.shape[0])
            cm += get_confusion_matrix(output, labels)
        test_accuracy = test_acc / len(test_loader)
        print(f"TEST ACCURACY ON {sdv}: ", test_accuracy)
        evaluation_results[f"{sdv}_acc"] = test_accuracy
        evaluation_results[f"{sdv}_cm"] = cm.tolist()

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
            extra_transforms=TRANSFORMS[t] if extra_transforms is None else transforms.Compose([TRANSFORMS[t], extra_transforms])
        )
        print("DATASET LOADED")
        test_acc = 0.0
        cm = torch.zeros(2, 2, dtype=torch.int64).to(device)
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.reshape((labels.shape[0])).to(device)
            output = torch.softmax(model(images), dim=1)
            test_acc += get_accuracy(output, labels, labels.shape[0])
            cm += get_confusion_matrix(output, labels)
        test_accuracy = test_acc / len(test_loader)
        print(f"TEST ACCURACY ON {t}: ", test_accuracy)
        evaluation_results[f"{t}_acc"] = test_accuracy
        evaluation_results[f"{t}_cm"] = cm.tolist()

    print()
    print("TESTING PERFORMANCE ON AUTOATTACK")
    for eps in LINF_EPSILONS:
        print("TESTING ON EPSILON: ", eps)
        train_loader, test_loader = load_dataset(
            real_path=REAL_LOC,
            fake_path=sdd_path[sdd_version],
            batch_size=32,
            samples=50000,
            size=size,
            no_transforms=True
        )
        print("DATASET LOADED")
        other_transforms, _ = get_transforms(size=size, already_tensor=True, extra_transforms=extra_transforms)

        def run_model_with_transforms(images):
            transformed = other_transforms(images)
            return model(transformed)
        adversary = AutoAttack(run_model_with_transforms, norm='Linf', eps=eps, version='custom', attacks_to_run=['apgd-ce'])

        test_acc = 0.0
        cm = torch.zeros(2, 2, dtype=torch.int64).to(device)
        for i, (images, labels) in (pbar := tqdm(enumerate(test_loader))):
            images = images.to(device)
            labels = labels.reshape((labels.shape[0])).to(device)
            x_pgm = adversary.run_standard_evaluation(images, labels, bs=32)
            output = torch.softmax(run_model_with_transforms(x_pgm), dim=1)
            test_acc += get_accuracy(output, labels, labels.shape[0])
            cm += get_confusion_matrix(output, labels)
            pbar.set_description(f"tentative acc: {test_acc / (i + 1)}")
            if i >= 20:
                break
        test_accuracy = test_acc / len(test_loader)
        print(f"TEST ACCURACY ON {eps}: ", test_accuracy)
        evaluation_results[f"linf_{eps}_acc"] = test_accuracy
        evaluation_results[f"linf_{eps}_cm"] = cm.tolist()

    with open(f'./{name}_partial_robustness.json', 'w') as f:
        f.write(json.dumps(evaluation_results))
