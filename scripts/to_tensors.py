import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset import XRayDataset
from utils.vocabulary import Vocabulary

def main():
    csv_in = "./data/labels.csv"
    img_root = "./data/"
    out_root = "./data/tensors"
    os.makedirs(out_root, exist_ok=True)

    df = pd.read_csv(csv_in)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    vocab = Vocabulary(min_freq=1)
    if "caption_tokens" not in df.columns:
        vocab.build(df["report"].astype(str).tolist())

    dataset = XRayDataset(
        df=df,
        root_dir=img_root,
        image_col="image_path",
        text_col="report",
        vocab=vocab,
        max_len=64,
        transform=tfm,
        use_gpu_loader=False, 
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
    )

    for i, (images, _) in enumerate(loader, 1):
        if i % 1000 == 0:
            print(f"{i}/{len(dataset)}")
        rel_path = dataset.df.loc[i - 1, "image_path"]
        out_path = os.path.join(out_root, rel_path + ".pt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(images[0], out_path)

if __name__ == "__main__":
    main()