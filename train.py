import torch
from utils.dataloader import ImageDataLoader
from models.cnn_lstm.model import CNNLSTM
from utils.trainer import Trainer, TrainConfig

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = ImageDataLoader(
        data_dir="./data/mimic-cxr",
        csv_file_name="labels.csv",
        image_col="image_path",
        text_col="report",
        batch_size=32,
        num_workers=1,
        max_len=64,
        min_freq=2,
    )
    train_loader, val_loader = data.get_loaders()

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
            epochs=50,
            unfreeze_epoch=15,
            lr_decoder=1e-3,
            lr_decoder_unfrozen=3e-4,
            lr_cnn=1e-5,
            weight_decay=1e-4,
        ),
    )
    trainer.train()

if __name__ == "__main__":
    main()