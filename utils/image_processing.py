"""Checkpoint-controlled image transforms for scientific figures."""

from PIL import Image, ImageOps
from torchvision import transforms


class FitAndPad:
    """Preserve the complete figure and its aspect ratio on a white square."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(image, (self.size, self.size),
                            method=Image.Resampling.BILINEAR, color="white")


def build_image_transform(size=256, preprocessing="crop"):
    if size < 32 or size % 32:
        raise ValueError("image_size must be a positive multiple of 32 (at least 32)")
    if preprocessing == "crop":
        spatial = [transforms.Resize(round(size * 288 / 256)),
                   transforms.CenterCrop((size, size))]
    elif preprocessing == "pad":
        spatial = [FitAndPad(size)]
    else:
        raise ValueError("image_preprocessing must be crop or pad")
    return transforms.Compose([*spatial, transforms.ToTensor()])
