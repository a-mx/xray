import argparse
import os
import torch

from utils.dataloader import ImageDataLoader
from models.cnn_lstm.model import CNNLSTM
from utils.trainer import Trainer, TrainConfig


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", type=str, default="./data/mimic-cxr")
    p.add_argument("--csv", type=str, default="labels_tokenized.csv")
    p.add_argument("--image-col", type=str, default="image_path")
    p.add_argument("--text-col", type=str, default="report")
    p.add_argument("--tensor-root", type=str, default="./data/mimic-cxr_tensors")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--max-len", type=int, default=48)
    p.add_argument("--min-freq", type=int, default=2)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--unfreeze-epoch", type=int, default=1)
    p.add_argument("--lr-decoder", type=float, default=1e-3)
    p.add_argument("--lr-decoder-unfrozen", type=float, default=3e-4)
    p.add_argument("--lr-cnn", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--ckpt", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = ImageDataLoader(
        data_dir=args.data_dir,
        csv_file_name=args.csv,
        image_col=args.image_col,
        text_col=args.text_col,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        min_freq=args.min_freq,
    )

    train_loader, val_loader = data.get_loaders()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)

        model = CNNLSTM(
            num_embeddings=ckpt["config"]["num_embeddings"],
            embedding_dim=ckpt["config"]["embedding_dim"],
            hidden_size=ckpt["config"]["hidden_size"],
            num_layers=ckpt["config"]["num_layers"],
            dropout=ckpt["config"]["dropout"],
            padding_idx=ckpt["config"]["padding_idx"],
            train_backbone=False,
        ).to(device)

        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("Loaded ckpt:", args.ckpt)
        print("missing:", missing)
        print("unexpected:", unexpected)
    else:
        model = CNNLSTM(
            num_embeddings=len(data.vocab.stoi),
            embedding_dim=256,
            hidden_size=512,
            num_layers=1,
            dropout=0.2,
            train_backbone=False,
            padding_idx=data.vocab.pad_idx,
        ).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=data.vocab,
        device=device,
        config=TrainConfig(
            epochs=args.epochs,
            unfreeze_epoch=args.unfreeze_epoch,
            lr_decoder=args.lr_decoder,
            lr_decoder_unfrozen=args.lr_decoder_unfrozen,
            lr_cnn=args.lr_cnn,
            weight_decay=args.weight_decay,
        ),
    )

    trainer.train()


if __name__ == "__main__":
    main()