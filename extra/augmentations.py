import torch
from PIL import Image, ImageEnhance
import copy
import numbers
from collections.abc import Sequence
from typing import Tuple, List
from torchvision.transforms import functional as F


class RandomHorizontalFlip(torch.nn.Module):
    def forward(self, img: Image.Image, p):
        if torch.rand(1) < 0.5:
            # draw_kps(img, p, "before.png")
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            p = p.clone()
            p[..., 0] = img.width - 1 - p[..., 0]
            p = p.clone()
            p[..., 0] = img.width - 1 - p[..., 0]
            # draw_kps(img, p, "after.png")
            # import pdb; pdb.set_trace()
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


def draw_kps(img, P, save):
    qi = img.copy()
    q = qi.load()
    for nx, ny in zip(P[..., 0], P[..., 1]):
        nx, ny = int(ny), int(nx)
        try:
            q[ny, nx] = (255, 0, 255)
            q[ny, nx + 1] = (255, 0, 255)
            q[ny, nx - 1] = (255, 0, 255)
            q[ny + 1, nx] = (255, 0, 255)
            q[ny - 1, nx] = (255, 0, 255)
        except IndexError:
            pass
    qi.save(save)


def _get_perspective_coeffs(
        startpoints: List[List[int]], endpoints: List[List[int]]
) -> List[float]:
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.

    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float)

    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    b_matrix = torch.tensor(startpoints, dtype=torch.float).view(8)
    res = torch.lstsq(b_matrix, a_matrix)[0]

    output: List[float] = res.squeeze(1).tolist()
    return output


class RandomPerspective(torch.nn.Module):
    """Performs a random perspective transformation of the given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float): probability of the image being transformed. Default is 0.5.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, distortion_scale=0.5, p=0.5, fill=0):
        super().__init__()
        self.p = p
        self.distortion_scale = distortion_scale

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    def forward(self, img, p):
        """
        Args:
            img (PIL Image or Tensor): Image to be Perspectively transformed.

        Returns:
            PIL Image or Tensor: Randomly transformed image.
        """

        fill = self.fill
        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if torch.rand(1) < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            a, b, c, d, e, f, g, h = _get_perspective_coeffs(endpoints, startpoints)
            x = p[..., 0]
            y = p[..., 1]
            xn = (a * x + b * y + c) / (g * x + h * y + 1)
            yn = (d * x + e * y + f) / (g * x + h * y + 1)
            pn = torch.stack([xn, yn], -1)
            return F.perspective(img, startpoints, endpoints, Image.BICUBIC, fill), pn
        return img, p

    @staticmethod
    def get_params(width: int, height: int, distortion_scale: float) -> Tuple[List[List[int]], List[List[int]]]:
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
