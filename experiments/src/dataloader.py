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
    path: str, batch_size: int, samples: int, spectogram: bool, compress: bool, size: int
):
    training_transform, testing_transform = get_transforms(spectogram, compress, size)
    dataset = StableDiffusionDetectionDataset(
        path, ".", transform=None, samples=samples
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    # Dont think this works, they use the same underlying dataset
    train_dataset.dataset.transform = training_transform
    test_dataset.dataset.transform = testing_transform

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader, test_loader

def load_evaluation_dataset(path: str, batch_size: int, spectogram: bool, compress: bool, size: int):
    _, testing_transform = get_transforms(spectogram, compress, size)
    dataset = ImageFolder(path, transform=testing_transform)
    image_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return image_loader


def get_transforms(spectogram: bool, compress: bool, size:int):
    transform = transforms.ToTensor()
    if compress:
        transform = transforms.Compose([transforms.Lambda(JPEGcompression), transform])
    if spectogram:
        transform = transforms.Compose([transform, transforms.Lambda(spec)])
    transform = transforms.Compose([transform, transforms.Resize((size, size))])

    training_transform = transforms.Compose(
        [transform, transforms.RandomHorizontalFlip()]
    )
    testing_transform = transform

    return training_transform, testing_transform


class StableDiffusionDetectionDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None,
        samples=None,
    ):
        self.stable_diffusion_detection_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        if samples is not None:
            small_true = self.stable_diffusion_detection_frame[
                self.stable_diffusion_detection_frame["label"] == 1
            ][:samples]
            small_false = self.stable_diffusion_detection_frame[
                self.stable_diffusion_detection_frame["label"] == 0
            ][:samples]

            self.stable_diffusion_detection_frame = pd.concat((small_true, small_false))

    def __len__(self):
        return len(self.stable_diffusion_detection_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.root_dir, self.stable_diffusion_detection_frame.iloc[idx, 0]
        )
        image = Image.open(img_name)
        label = self.stable_diffusion_detection_frame.iloc[idx, 1]
        label = np.array([label])

        if self.transform:
            image = self.transform(image)
        return image, label
