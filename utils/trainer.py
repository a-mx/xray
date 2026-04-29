import torch
import torch.nn as nn
from torch.optim import AdamW


class TrainConfig:
    def __init__(
        self,
        epochs=10,
        unfreeze_epoch=4,
        lr_decoder=1e-3,
        lr_decoder_unfrozen=5e-4,
        lr_cnn=1e-5,
        weight_decay=1e-4,
        max_gen_len=30,
        save_path="best_cnn_lstm.pt",
    ):
        self.epochs = epochs
        self.unfreeze_epoch = unfreeze_epoch
        self.lr_decoder = lr_decoder
        self.lr_decoder_unfrozen = lr_decoder_unfrozen
        self.lr_cnn = lr_cnn
        self.weight_decay = weight_decay
        self.max_gen_len = max_gen_len
        self.save_path = save_path


class Trainer:
    def __init__(self, model, train_loader, val_loader, vocab, device, config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        self.config = config

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx)
        self.optimizer = self.build_optimizer(
            lr_decoder=self.config.lr_decoder,
            lr_cnn=self.config.lr_cnn,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=1,
        )

        self.best_val = float("inf")

    def set_backbone_trainable(self, trainable: bool):
        for p in self.model.cnn.encoder.parameters():
            p.requires_grad = trainable

    def build_optimizer(self, lr_decoder=1e-3, lr_cnn=1e-5, weight_decay=1e-4):
        cnn_params = [p for p in self.model.cnn.encoder.parameters() if p.requires_grad]
        other_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and not n.startswith("cnn.encoder")
        ]

        param_groups = [
            {"params": other_params, "lr": lr_decoder, "weight_decay": weight_decay},
        ]

        if len(cnn_params) > 0:
            param_groups.append(
                {"params": cnn_params, "lr": lr_cnn, "weight_decay": weight_decay}
            )

        return AdamW(param_groups)

    def run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss = 0.0

        for images, captions in loader:
            images = images.to(self.device, non_blocking=True)
            captions = captions.to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad()

            logits = self.model(images, captions)
            targets = captions[:, 1:]
            loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def save_checkpoint(self):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vocab_stoi": self.vocab.stoi,
                "config": {
                    "embed_size": 256,
                    "hidden_size": 512,
                    "num_layers": 1,
                    "dropout": 0.2,
                    "pad_idx": self.vocab.pad_idx,
                },
            },
            self.config.save_path,
        )

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            if epoch == self.config.unfreeze_epoch:
                self.set_backbone_trainable(True)
                self.optimizer = self.build_optimizer(
                    lr_decoder=self.config.lr_decoder_unfrozen,
                    lr_cnn=self.config.lr_cnn,
                    weight_decay=self.config.weight_decay,
                )

            train_loss = self.run_epoch(self.train_loader, train=True)
            val_loss = self.run_epoch(self.val_loader, train=False)
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f}")

            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint()
                print(f"Saved {self.config.save_path}")