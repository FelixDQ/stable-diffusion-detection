import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import requests
from io import BytesIO
import os

class StableDiffusionDetectionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, samples=None):
        self.stable_diffusion_detection_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        if samples is not None:
            small_true = self.stable_diffusion_detection_frame[self.stable_diffusion_detection_frame['label'] == 1][:samples]
            small_false = self.stable_diffusion_detection_frame[self.stable_diffusion_detection_frame['label'] == 0][:samples]

            self.stable_diffusion_detection_frame = pd.concat((small_true, small_false))

    def __len__(self):
        return len(self.stable_diffusion_detection_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.stable_diffusion_detection_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.stable_diffusion_detection_frame.iloc[idx, 1]
        label = np.array([label])

        if self.transform:
            image = self.transform(image)
        return image, label
    
def flat_map(f, xs):
  ys = []
  for x in xs:
      ys.extend([f(x) for x in x])
  return ys

def get_img_files(loc):
    return sorted(flat_map(lambda x: os.path.join(x[0], x[1]), [[(root, file) for file in files if file.endswith('.png') or file.endswith('.jpg')] for root, dirs, files in os.walk(loc)]))

class SDDDataset(Dataset):
    def __init__(self, fake_loc, real_loc, transform=None, samples=None, skip=0):
        import os


        fake_files = get_img_files(fake_loc)
        real_files = get_img_files(real_loc)
        
        self.transform = transform

        if samples is not None:
            fake_files = fake_files[skip:skip+samples]
            real_files = real_files[skip:skip+samples]
            
        df = pd.DataFrame({'file': fake_files + real_files, 'label': [0] * len(fake_files) + [1] * len(real_files)})

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