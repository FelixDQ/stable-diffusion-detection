from src.models import get_inception_model, get_xception_model, get_densenet_model, get_vit_model
from src.experiment import run_experiment
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

run_experiment(get_densenet_model, "vit", spectogram=True, compress=True, size=299)
