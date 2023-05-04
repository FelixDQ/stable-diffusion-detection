from src.models import get_convnext_model, get_xception_model, get_vit_model
from src.experiment import run_experiment, sdd_path
from src.robustness import test_robustness
import logging
from torchvision import transforms

from src.util import rand_noise, rand_pad, squeeze

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

choice = transforms.RandomChoice(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(rand_pad),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.001, 5)),
        transforms.Lambda(rand_noise),
        transforms.RandomAffine(degrees=(0, 180), translate=(0, 0.15)),
        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    ]
)

tmp_test_models = {
    "vit": get_vit_model,
}

tmp_test_transforms = [
    "baseline",
    "horizontal_flip",
    "pad",
    "resize_crop",
    "colorjitter",
    "blur",
    "noise",
    "affine",
    "grayscale",
    "perspective",
]

if __name__ == "__main__":

    print("Starting array mode")
    combinations = list(itertools.product(models.keys(), sdd_path.keys(), [None, "adversarial_rando", "transforms_choice", "squeezed"]))
    print(combinations)
    for model, sdd_version, model_suffix in combinations:
        print(f"Running {model} {sdd_version}")
        print("Testing robustness")
        test_robustness(
            models[model],
            model,
            size=model_size[model],
            sdd_version=sdd_version,
            model_suffix=model_suffix,
        )

