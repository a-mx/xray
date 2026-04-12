import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Tuple, Optional
from torch import Generator
from PIL import Image

class XRayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, image_dir:str, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.image_base_dir = os.path.join(root_dir, image_dir) if image_dir else root_dir


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_base_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

class ImageDataLoader:
    def __init__(
        self, 
        data_dir: str, 
        csv_file_name: str,
        image_dir: str = None,
        batch_size: int = 32, 
        image_height: int = 80,
        image_width: int = 320,
        val_split: float = 0.2,
        num_workers: int = 0 
    ):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, csv_file_name)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.val_split = val_split
        self.num_workers = num_workers
        
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        
        train_transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.image_height, self.image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        return train_transform, val_transform

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Nie znaleziono pliku CSV: {self.csv_path}")
        df = pd.read_csv(self.csv_path, index_col=0)

        train_transform, val_transform = self.get_transforms()

        dataset_size = len(df)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size

        indices = list(range(dataset_size))
        full_dataset_dummy = range(dataset_size) 
        generator = Generator().manual_seed(0)
        train_indices_subset, val_indices_subset = random_split(full_dataset_dummy, [train_size, val_size], generator=generator)
        
        train_idx = train_indices_subset.indices
        val_idx = val_indices_subset.indices

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = XRayDataset(df=train_df, root_dir=self.data_dir, image_dir=self.image_dir, transform=train_transform)
        val_dataset = XRayDataset(df=val_df, root_dir=self.data_dir, image_dir=self.image_dir, transform=val_transform)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        return train_loader, val_loader

if __name__ == "__main__":
    loader = ImageDataLoader(data_dir="./data/iu", image_dir="images/images_normalized",csv_file_name="labels.csv", batch_size=40)
    try:
        train_dl, val_dl = loader.get_loaders()
        images, data = next(iter(train_dl))
        print(f"Shape: {images.shape}")
        print(f"Data: {data}")
    except Exception as e:
        print(f"Exception: {e}")
        