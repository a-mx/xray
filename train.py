import argparse
import os
import torch

from utils.dataloader import ImageDataLoader
from models.cnn_lstm.model import CNNLSTM
from models.cnn_transformer.model import CNNTransformerDecoder
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

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--unfreeze-epoch", type=int, default=20)
    p.add_argument("--lr-decoder", type=float, default=1e-3)
    p.add_argument("--lr-decoder-unfrozen", type=float, default=3e-4)
    p.add_argument("--lr-cnn", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--model-type", type=str, default="cnn_lstm", choices=["cnn_lstm", "cnn_transformer"])
    p.add_argument("--nhead", type=int, default=8)

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

    def build_model(model_type: str, config: dict | None = None):
        config = config or {}

        if model_type == "cnn_transformer":
            return CNNTransformerDecoder(
                num_embeddings=config.get("num_embeddings", len(data.vocab.stoi)),
                embedding_dim=config.get("embedding_dim", 256),
                hidden_size=config.get("hidden_size", 512),
                num_layers=config.get("num_layers", 2),
                nhead=config.get("nhead", args.nhead),
                dropout=config.get("dropout", 0.2),
                padding_idx=config.get("padding_idx", data.vocab.pad_idx),
                max_len=config.get("max_len", args.max_len),
                train_backbone=False,
            ).to(device)

        return CNNLSTM(
            num_embeddings=config.get("num_embeddings", len(data.vocab.stoi)),
            embedding_dim=config.get("embedding_dim", 256),
            hidden_size=config.get("hidden_size", 512),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.2),
            padding_idx=config.get("padding_idx", data.vocab.pad_idx),
            train_backbone=False,
        ).to(device)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        ckpt_config = ckpt["config"]
        model_type = ckpt_config.get("model_type")
        if model_type is None:
            model_type = "cnn_transformer" if "nhead" in ckpt_config else "cnn_lstm"

        model = build_model(model_type, ckpt_config)

        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("Loaded ckpt:", args.ckpt)
        print("model_type:", model_type)
        print("missing:", missing)
        print("unexpected:", unexpected)
    else:
        model = build_model(args.model_type)

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