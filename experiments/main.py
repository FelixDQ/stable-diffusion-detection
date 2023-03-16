from src.models import get_convnext_model, get_xception_model, get_vit_model
from src.experiment import run_experiment, sdd_path
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import itertools
import numpy as np
import os

models = {
    "xception": get_xception_model,
    "vit": get_vit_model,
    "convnext": get_convnext_model,
}

model_size = {
    "xception": 299,
    "convnext": 224,
    "vit": 224,
}

if __name__ == "__main__":
    if len(sys.argv) == 1 and "SLURM_ARRAY_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        print("Starting array mode")
        combinations = list(itertools.product(models.keys(), sdd_path.keys()))
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_count = os.environ["SLURM_ARRAY_TASK_COUNT"]
        splits = np.array_split(combinations, int(task_count))
        print(combinations)
        combinations_in_this_split = splits[task_id-1]
        print(combinations_in_this_split)
        for model, sdd_version in combinations_in_this_split:
            print(f"Running {model} {sdd_version}")
            run_experiment(models[model], model, size=model_size[model], sdd_version=sdd_version)
    else:
        try:
            model, sdd_version = sys.argv[1:]
            if model not in models:
                raise ValueError(f"Model {model} not found.")
            if sdd_version not in ["1.4", "2.0", "2.1"]:
                raise ValueError(f"SDD version {sdd_version} not found.")

        except Exception as e:
            logging.error(e)
            print("Usage: python main.py <model> <sdd_version>")
            print(" model: xception, convnext, vit")
            print(" sdd_version: 1.4, 2.0, 2.1")
            print("Example: python main.py xception 2.1")
            sys.exit(1)

        run_experiment(models[model], model, size=model_size[model], sdd_version=sdd_version)
