import torch
from PIL import Image


class RandomHorizontalFlip:
    def forward(self, img: Image.Image, p):
        if torch.rand(1) < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            p[..., 0] = img.width - p[..., 0]
        return img, p
