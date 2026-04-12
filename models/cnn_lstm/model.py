import torch
import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return x