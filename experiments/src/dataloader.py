from torchvision import transforms
from torchvision.datasets import ImageFolder
from src.util import spec, JPEGcompression
import torch
import pandas as pd
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
import numpy as np


def load_dataset(
    real_path: str, fake_path: str, batch_size: int, samples: int, size: int, extra_transforms=None, no_transforms=False
):
    train_size = int(0.8 * samples)
    test_size = samples - train_size

    training_transform, testing_transform = get_transforms(size, extra_transforms, no_transforms)

    train_dataset = SDDDataset(
        fake_path, real_path, transform=training_transform, samples=train_size
    )
    test_dataset = SDDDataset(
        fake_path,
        real_path,
        transform=testing_transform,
        samples=test_size,
        skip=train_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=16
    )

    return train_loader, test_loader


def get_transforms(size: int, extra_transforms=None, no_transforms=False):
    transform = transforms.ToTensor()

    if no_transforms:
        return transform, transform

    if extra_transforms:
        transform = transforms.Compose([extra_transforms, transform])



    transform = transforms.Compose(
        [
            transform,
            transforms.Resize((size, size), antialias=None),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    training_transform = transforms.Compose([transform])
    testing_transform = transform

    return training_transform, testing_transform


def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend([f(x) for x in x])
    return ys


def get_img_files(loc):
    return sorted(
        flat_map(
            lambda x: os.path.join(x[0], x[1]),
            [
                [
                    (root, file)
                    for file in files
                    if file.endswith(".png") or file.endswith(".jpg")
                ]
                for root, dirs, files in os.walk(loc)
            ],
        )
    )


class SDDDataset(Dataset):
    def __init__(self, fake_loc, real_loc, transform=None, samples=None, skip=0):
        import os

        fake_files = get_img_files(fake_loc)
        real_files = get_img_files(real_loc)

        self.transform = transform

        if samples is not None:
            fake_files = fake_files[skip : skip + samples]
            real_files = real_files[skip : skip + samples]

        df = pd.DataFrame(
            {
                "file": fake_files + real_files,
                "label": [0] * len(fake_files) + [1] * len(real_files),
            }
        )

        self.stable_diffusion_detection_frame = df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.stable_diffusion_detection_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.stable_diffusion_detection_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.stable_diffusion_detection_frame.iloc[idx, 1]
        label = np.array([label])

        if self.transform:
            image = self.transform(image)
        return image, label
