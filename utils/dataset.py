import os
from typing import Optional
from .vocabulary import Vocabulary
from .imageloader import GPUImageLoader
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import ast

class XRayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        image_col: str,
        text_col: str,
        vocab: Vocabulary,
        max_len: Optional[int] = None,
        transform=None,
        use_gpu_loader: bool = True,
        tensor_root: Optional[str] = None
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.image_col = image_col
        self.text_col = text_col
        self.vocab = vocab
        self.max_len = max_len
        self.tensor_root = tensor_root

        if use_gpu_loader:
            self.loader = GPUImageLoader()
        else:
            self.loader = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel_path = self.df.loc[idx, self.image_col]
        img_path = os.path.join(self.root_dir, img_rel_path)

        tensor_path = None
        if self.tensor_root is not None:
            tensor_path = os.path.join(self.tensor_root, img_rel_path + ".pt")

        if tensor_path and os.path.exists(tensor_path):
            image = torch.load(tensor_path, map_location="cpu")
        elif self.loader:
            image = self.loader(img_path)
        else:
            if self.transform is None:
                raise RuntimeError(f"Missing tensor and no transform for: {img_rel_path}")
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)

        if "caption_tokens" in self.df.columns:
            tokens = ast.literal_eval(self.df.loc[idx, "caption_tokens"])
            caption_ids = torch.tensor(tokens, dtype=torch.long)
        else:
            caption_text = self.df.loc[idx, self.text_col]
            caption_ids = torch.tensor(
                self.vocab.encode(caption_text, max_len=self.max_len),
                dtype=torch.long,
            )
        return image, caption_ids