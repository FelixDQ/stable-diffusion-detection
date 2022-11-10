from src.models import get_inception_model
from src.experiment import run_experiment
import logging
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

run_experiment(get_inception_model, "inception", spectogram=True, compress=False, size=299)


