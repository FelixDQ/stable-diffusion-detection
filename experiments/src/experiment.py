from src.dataloader import load_dataset, load_evaluation_dataset
from src.train import train_model
from src.models import get_inception_model
from src.evaluate import evaluate_model
from src.util import get_device
from torchvision import models
from torch import nn
import torch

evaluation_datasets = ["evaluation/midjourney", "evaluation/beaches"]


def run_experiment(model_func, model_name: str, spectogram: bool, compress: bool, size: int):
    name = f"{model_name}" + "_spectogram" if spectogram else "" + "_compressed" if compress else ""
    print("RUNNING EXPERIMENT FOR: ", name)
    device = get_device()

    train_loader, test_loader = load_dataset(
        path="data/stable_diffusion_detection.csv",
        batch_size=2,
        samples=8,
        spectogram=spectogram,
        compress=compress,
        size=size,
    )

    # TRAINING
    model = model_func()
    model.to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)

    training_acc, test_acc = train_model(
        model=model,
        optimizer=optimizer,
        criterion=nn.BCELoss(),
        device=device,
        epochs=17,
        train_loader=train_loader,
        test_loader=test_loader,
    )
    torch.save(model.state_dict(), f"./{name}.pt")

    # EVALUATION
    evaluation_results = {
        'training_acc': training_acc,
        'test_acc': test_acc,
    }
    for path in evaluation_datasets:
        # model = model_func()
        # model.load_state_dict(torch.load(f"./{name}.pt"))
        # model.to(device)

        evaluation_loader = load_evaluation_dataset(
            path=path, batch_size=2, spectogram=spectogram, compress=compress, size=size
        )
        cm = evaluate_model(model, evaluation_loader, device)
        evaluation_results[path + "_cm"] = cm
        accuacy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        evaluation_results[path + "_accuracy"] = accuacy

    print(evaluation_results)
