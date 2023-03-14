from src.models import get_inception_model, get_xception_model, get_densenet_model, get_vit_model
from src.experiment import run_experiment
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys

models = {
    "inception": get_inception_model,
    "xception": get_xception_model,
    "densenet": get_densenet_model,
    "vit": get_vit_model,
}

model_size = {
    "inception": 299,
    "xception": 299,
    "densenet": 299,
    "vit": 224,
}

if __name__ == "__main__":
    try:
        model = sys.argv[1]
        if model not in models:
            raise ValueError(f"Model {model} not found.")



    except Exception as e:
        logging.error(e)
        print("Usage: python main.py <model> <spectogram> <compress>")
        print(" model: inception, xception, densenet, vit")
        print(" spectogram: 1, 0")
        print(" compress: 1, 0")
        print("Example: python main.py inception 1 0")
        sys.exit(1)

    run_experiment(models[model], model, size=model_size[model])
