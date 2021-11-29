import torch
from PIL import Image, ImageEnhance


class RandomHorizontalFlip(torch.nn.Module):
    def forward(self, img: Image.Image, p):
        if torch.rand(1) < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            p = p.copy()
            p[..., 0] = img.width - 1 - p[..., 0]
        return img, p


class RandomAdjustSharpness(torch.nn.Module):
    def __init__(self, p=0.4) -> None:
        super().__init__()
        self.p = p

    def forward(self, img: Image.Image):
        val = torch.rand(1).item() * 2
        while torch.rand(1) < self.p:
            return self.adjust_sharpness(img, val)
        return img

    def adjust_sharpness(self, img, sharpness_factor):
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness_factor)
        return img
