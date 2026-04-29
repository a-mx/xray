import os
from typing import Optional
from .vocabulary import Vocabulary
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class XRayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        image_dir: Optional[str],
        image_col: str,
        text_col: str,
        vocab: Vocabulary,
        max_len: Optional[int] = None,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.image_col = image_col
        self.text_col = text_col
        self.vocab = vocab
        self.max_len = max_len
        self.image_base_dir = os.path.join(root_dir, image_dir) if image_dir else root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel_path = self.df.loc[idx, self.image_col]
        img_path = os.path.join(self.image_base_dir, img_rel_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption_text = self.df.loc[idx, self.text_col]
        caption_ids = torch.tensor(
            self.vocab.encode(caption_text, max_len=self.max_len),
            dtype=torch.long,
        )
        return image, caption_ids