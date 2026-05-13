import torch
from torchvision.io import read_file, decode_jpeg
from torchvision.transforms.functional import resize, normalize

class GPUImageLoader:
    def __init__(self, target_size=224, mean=None, std=None):
        self.target_size = target_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(self, path):
        data = read_file(path)
        img = decode_jpeg(data, device="cuda") 
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        img = resize(img, [self.target_size, self.target_size], antialias=True)
        img = img.float() / 255.0
        img = normalize(img, self.mean, self.std)
        return img