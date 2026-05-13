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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.backbone_out = densenet.classifier.in_features
        self.linear = nn.Linear(self.backbone_out, embedding_dim)

        if not train_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
        train_backbone: bool = False,
        padding_idx: int = 0,
        cnn_weights: Optional[DenseNet121_Weights] = DenseNet121_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        self.padding_idx=padding_idx
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cnn = CNN(embedding_dim, train_backbone, cnn_weights)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.img_proj = nn.Linear(2*embedding_dim, embedding_dim)
        
        self.dropout = dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=self.dropout)

        self.init_h = nn.Linear(embedding_dim, hidden_size)
        self.init_c = nn.Linear(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_embeddings)

        self.config = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "train_backbone": train_backbone,
            "padding_idx": padding_idx,
            "cnn_weights": str(cnn_weights) if cnn_weights is not None else None,
        }

    def init(self, embedding: torch.Tensor):
        h0 = torch.tanh(self.init_h(embedding)).unsqueeze(0)
        c0 = torch.tanh(self.init_c(embedding)).unsqueeze(0)
        if self.num_layers > 1:
            h0 = h0.repeat(self.num_layers, 1, 1)
            c0 = c0.repeat(self.num_layers, 1, 1)
        return h0, c0

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        img_embed = self.cnn(images)
        h0, c0 = self.init(img_embed)
        x = self.embedding(captions[:, :-1])
        img_exp = img_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        lstm_in = torch.cat([x, img_exp], dim=2)
        lstm_in = self.img_proj(lstm_in)
        out, _ = self.lstm(lstm_in, (h0, c0))
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits
    
    @torch.no_grad()
    def generate(self, images, start_token_id, end_token_id, max_len=30):
        self.eval()
        img_embed = self.cnn(images)
        batch_size = images.size(0)
        cur = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=images.device)
        h, c = self.init(img_embed)

        outputs = []
        for _ in range(max_len):
            x = self.embedding(cur)
            img_exp = img_embed.unsqueeze(1)
            concat = torch.cat([x, img_exp], dim=2)
            lstm_input = self.img_proj(concat)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            logits = self.classifier(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            outputs.append(next_token)
            cur = next_token
        return torch.cat(outputs, dim=1)

if __name__ == "__main__":
    pass