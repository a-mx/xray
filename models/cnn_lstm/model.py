import torch
import torch.nn as nn
from typing import Optional
from torchvision import models
from torchvision.models import ResNet50_Weights

class CNN(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        train_backbone: bool = False,
        weights: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V2,
    ):
        super().__init__()
        resnet = models.resnet50(weights=weights)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone_out = resnet.fc.in_features
        self.linear = nn.Linear(self.backbone_out, embedding_size)

        if not train_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x