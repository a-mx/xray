import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from torchvision.models import DenseNet121_Weights

class CNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        train_backbone: bool = False,
        weights: Optional[DenseNet121_Weights] = DenseNet121_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        densenet = models.densenet121(weights=weights)
        self.features = densenet.features
        self.backbone_out = densenet.classifier.in_features
        self.linear = nn.Linear(self.backbone_out, embedding_dim)
        if not train_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = x.flatten(2).transpose(1, 2)
        x = self.linear(x)
        return x

class CNNTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.2,
        train_backbone: bool = False,
        padding_idx: int = 0,
        max_len: int = 48,
        cnn_weights: Optional[DenseNet121_Weights] = DenseNet121_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.cnn = CNN(embedding_dim, train_backbone, cnn_weights)

        self.token_embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.pos_embed = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embedding_dim, num_embeddings)

        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.config = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "nhead": nhead,
            "dropout": dropout,
            "train_backbone": train_backbone,
            "padding_idx": padding_idx,
            "max_len": max_len,
            "cnn_weights": str(cnn_weights) if cnn_weights is not None else None,
        }

    def causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        img_embed = self.cnn(images)
        memory = img_embed

        tgt = captions[:, :-1]
        t = tgt.size(1)
        pos = torch.arange(t, device=tgt.device).unsqueeze(0)
        tgt_emb = self.token_embed(tgt) + self.pos_embed(pos)
        tgt_emb = self.dropout(tgt_emb)

        tgt_mask = self.causal_mask(t, tgt.device)
        tgt_pad = tgt.eq(self.padding_idx)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
        )
        return self.classifier(out)

    @torch.no_grad()
    def generate(self, images, start_token_id, end_token_id, max_len=48):
        self.eval()
        img_embed = self.cnn(images)
        memory = img_embed

        batch_size = images.size(0)
        cur = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=images.device)

        outputs = []
        for _ in range(max_len):
            t = cur.size(1)
            if t > self.pos_embed.num_embeddings:
                break
            pos = torch.arange(t, device=cur.device).unsqueeze(0)
            tgt_emb = self.token_embed(cur) + self.pos_embed(pos)
            tgt_emb = self.dropout(tgt_emb)

            tgt_mask = self.causal_mask(t, cur.device)
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.classifier(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            outputs.append(next_token)
            cur = torch.cat([cur, next_token], dim=1)

            if torch.all(next_token == end_token_id):
                break

        tokens = torch.cat(outputs, dim=1)
        return tokens
if __name__ == "__main__":
    model = CNNTransformerDecoder(num_embeddings=5000)
    img = torch.randn(2, 3, 224, 224)
    cap = torch.randint(0, 5000, (2, 10))
    logits = model(img, cap)
    print(logits.shape)
    gen = model.generate(img, start_token_id=1, end_token_id=2, max_len=48)
    print(gen.shape)