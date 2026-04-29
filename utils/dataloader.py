import os
import os
from typing import Tuple, Optional
from .vocabulary import Vocabulary
from .dataset import XRayDataset
import pandas as pd
import torch
from torch import Generator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

class ImageDataLoader:
    def __init__(
        self,
        data_dir: str,
        csv_file_name: str,
        image_col: str = "image",
        text_col: str = "caption",
        image_dir: Optional[str] = None,
        batch_size: int = 32,
        image_height: int = 224,
        image_width: int = 224,
        val_split: float = 0.2,
        num_workers: int = 0,
        min_freq: int = 1,
        max_len: Optional[int] = 64,
    ):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, csv_file_name)
        self.image_col = image_col
        self.text_col = text_col
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.val_split = val_split
        self.num_workers = num_workers
        self.max_len = max_len

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.vocab = Vocabulary(min_freq=min_freq)

    def get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_height, self.image_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((self.image_height, self.image_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        return train_transform, val_transform

    def collate_fn(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.vocab.pad_idx)
        return images, captions

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Nie znaleziono pliku CSV: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if self.image_col not in df.columns or self.text_col not in df.columns:
            raise ValueError(
                f"CSV musi mieć kolumny '{self.image_col}' i '{self.text_col}'. "
                f"Dostępne: {list(df.columns)}"
            )

        self.vocab.build(df[self.text_col].astype(str).tolist())
        train_transform, val_transform = self.get_transforms()

        dataset_size = len(df)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size

        generator = Generator().manual_seed(0)
        full_dataset_dummy = range(dataset_size)
        train_subset, val_subset = random_split(
            full_dataset_dummy, [train_size, val_size], generator=generator
        )

        train_df = df.iloc[train_subset.indices].reset_index(drop=True)
        val_df = df.iloc[val_subset.indices].reset_index(drop=True)

        train_dataset = XRayDataset(
            df=train_df,
            root_dir=self.data_dir,
            image_dir=self.image_dir,
            image_col=self.image_col,
            text_col=self.text_col,
            vocab=self.vocab,
            max_len=self.max_len,
            transform=train_transform,
        )
        val_dataset = XRayDataset(
            df=val_df,
            root_dir=self.data_dir,
            image_dir=self.image_dir,
            image_col=self.image_col,
            text_col=self.text_col,
            vocab=self.vocab,
            max_len=self.max_len,
            transform=val_transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return train_loader, val_loader


if __name__ == "__main__":
    loader = ImageDataLoader(
        data_dir="./data/iu",
        image_dir="images/images_normalized",
        csv_file_name="labels.csv",
        image_col="filename",
        text_col="findings",
        batch_size=4,
    )
    train_dl, val_dl = loader.get_loaders()
    images, captions = next(iter(train_dl))
    print(f"Images shape: {images.shape}")
    print(f"Captions shape: {captions.shape}")
    print(f"Vocab size: {len(loader.vocab.stoi)}")
    print(f"pad/bos/eos: {loader.vocab.pad_idx}/{loader.vocab.bos_idx}/{loader.vocab.eos_idx}")
