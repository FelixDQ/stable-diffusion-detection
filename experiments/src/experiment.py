from src.dataloader import load_dataset, load_evaluation_dataset
from src.train import train_model
from src.models import get_inception_model
from src.evaluate import evaluate_model
from src.util import get_device
from torchvision import models
from torch import nn
import torch
import json

# evaluation_datasets = ["evaluation/midjourney", "evaluation/beaches"]
evaluation_datasets = [
    "/home/data_shares/sdd/datasets/beaches",
    "/home/data_shares/sdd/datasets/lexica",
    "/home/data_shares/sdd/datasets/midjourney",
]


def run_experiment(
    model_func, model_name: str, spectogram: bool, compress: bool, size: int
):
    name = (
        f"{model_name}"
        + ("_spectogram" if spectogram else "")
        + ("_compressed" if compress else "")
    )
    print("RUNNING EXPERIMENT FOR: ", name)
    device = get_device()

    train_loader, test_loader = load_dataset(
        path="/home/data_shares/sdd/stable_diffusion_detection.csv",
        batch_size=64,
        samples=50000,
        spectogram=spectogram,
        compress=compress,
        size=size,
    )

    # TRAINING
    model, optimizer, epochs = model_func()
    model.to(device)

    inception = model_name == "inception"

    training_acc, test_acc = train_model(
        model=model,
        optimizer=optimizer,
        criterion=nn.BCELoss(),
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
    for path in evaluation_datasets:
        # model = model_func()
        # model.load_state_dict(torch.load(f"./{name}.pt"))
        # model.to(device)

        evaluation_loader = load_evaluation_dataset(
            path=path, batch_size=2, spectogram=spectogram, compress=compress, size=size
        )
        cm = evaluate_model(model, evaluation_loader, device)
        print(cm)
        evaluation_results[path + "_cm"] = cm.tolist()
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        evaluation_results[path + "_accuracy"] = accuracy

    with open(f"./{name}.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print("FINISHED EXPERIMENT FOR: ", name)
