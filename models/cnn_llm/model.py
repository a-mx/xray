import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from torchvision.models import DenseNet121_Weights
from transformers import GPT2Config, GPT2LMHeadModel

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

class CNNLLM(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        train_backbone: bool = False,
        padding_idx: int = 0,
        max_len: int = 48,
        prefix_len: int = 49,
        cnn_weights: Optional[DenseNet121_Weights] = DenseNet121_Weights.IMAGENET1K_V1,
        freeze_llm: bool = False,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.prefix_len = prefix_len

        self.cnn = CNN(embedding_dim, train_backbone, cnn_weights)
        self.img_proj = nn.Linear(embedding_dim, embedding_dim)

        cfg = GPT2Config(
            vocab_size=num_embeddings,
            n_embd=embedding_dim,
            n_layer=num_layers,
            n_head=nhead,
            n_positions=max_len + prefix_len + 2,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            bos_token_id=1,
            eos_token_id=2,
        )
        self.llm = GPT2LMHeadModel(cfg)

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        self.config = {
            "model_type": "cnn_llm",
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "hidden_size": embedding_dim,
            "num_layers": num_layers,
            "nhead": nhead,
            "dropout": dropout,
            "train_backbone": train_backbone,
            "padding_idx": padding_idx,
            "max_len": max_len,
            "prefix_len": prefix_len,
            "cnn_weights": str(cnn_weights) if cnn_weights is not None else None,
        }

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        prefix = self.img_proj(self.cnn(images))
        tokens = captions[:, :-1]

        tok_emb = self.llm.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)

        text_attn = tokens.ne(self.padding_idx).long()
        prefix_attn = torch.ones(
            (tokens.size(0), prefix.size(1)),
            dtype=text_attn.dtype,
            device=images.device,
        )
        attn = torch.cat([prefix_attn, text_attn], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
        )
        logits = outputs.logits[:, prefix.size(1):, :]
        return logits

    @torch.no_grad()
    def generate(self, images, start_token_id, end_token_id, max_len=48):
        self.eval()
        prefix = self.img_proj(self.cnn(images))
        batch_size = images.size(0)
        cur = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=images.device,
        )

        outputs = []
        for _ in range(max_len):
            tok_emb = self.llm.get_input_embeddings()(cur)
            inputs_embeds = torch.cat([prefix, tok_emb], dim=1)

            attn = torch.ones(
                (batch_size, inputs_embeds.size(1)),
                dtype=torch.long,
                device=images.device,
            )
            logits = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                use_cache=False,
            ).logits

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            outputs.append(next_token)
            cur = torch.cat([cur, next_token], dim=1)

            if torch.all(next_token == end_token_id):
                break

        return torch.cat(outputs, dim=1)