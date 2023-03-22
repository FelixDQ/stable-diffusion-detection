import torch
from src.dataloader import load_dataset
from src.experiment import sdd_path, REAL_LOC
from src.util import get_accuracy, get_device

CH_EPSILONS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48]

def test_robustness(
    model_func, model_name: str, size: int, sdd_version : str
    ):
    name = (
        f"{model_name}_{sdd_version}"
    )

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
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.reshape((labels.shape[0])).to(device)
            output = torch.softmax(model(images), dim=1)
            test_acc += get_accuracy(output, labels, labels.shape[0])
        test_accuracy = test_acc / len(test_loader)
        print(f"TEST ACCURACY ON {sdv}: ", test_accuracy)
        evaluation_results[sdv] = test_accuracy

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
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            # CH attack
            labels = labels.reshape((labels.shape[0])).to(device)
            output = torch.softmax(model(images), dim=1)
            test_acc += get_accuracy(output, labels, labels.shape[0])
        test_accuracy = test_acc / len(test_loader)
        print(f"TEST ACCURACY ON {eps}: ", test_accuracy)
        evaluation_results[eps] = test_accuracy

