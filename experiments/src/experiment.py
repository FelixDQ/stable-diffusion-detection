from src.dataloader import load_dataset
from src.train import train_model
from src.models import get_inception_model
from src.evaluate import evaluate_model
from src.util import get_device
from torchvision import models
from torch import nn
import torch
import json

REAL_LOC = "/home/data_shares/sdd/data"

SDD14_LOC = "/home/data_shares/sdd/sdd1_4_results"
SDD20_LOC = "/home/data_shares/sdd/sdd2_0_results"
SDD21_LOC = "/home/data_shares/sdd/sdd2_1_results"

sdd_path = {
    "1.4": SDD14_LOC,
    "2.0": SDD20_LOC,
    "2.1": SDD21_LOC,
}

def run_experiment(
    model_func, model_name: str, size: int, sdd_version : str
):
    name = (
        f"{model_name}_{sdd_version}"
    )

    print("RUNNING EXPERIMENT FOR: ", name)
    device = get_device()

    train_loader, test_loader = load_dataset(
        real_path=REAL_LOC,
        fake_path=sdd_path[sdd_version],
        batch_size=64,
        samples=50000,
        size=size,
    )
    print("DATASET LOADED")

    # TRAINING
    model, optimizer, epochs = model_func()
    model.to(device)

    inception = model_name == "inception"
    print("STARTING TRAINING")
    training_acc, test_acc = train_model(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        inception=inception,
    )
    torch.save(model.state_dict(), f"./{name}.pt")

    # EVALUATION
    evaluation_results = {
        "training_acc": float(training_acc.view(-1).cpu().detach().numpy()[0]),
        "test_acc": float(test_acc.view(-1).cpu().detach().numpy()[0]),
    }

    with open(f"./{name}.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print("FINISHED EXPERIMENT FOR: ", name)
