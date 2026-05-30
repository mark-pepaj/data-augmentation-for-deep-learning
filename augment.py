import torchvision.transforms as transforms
from dataclasses import dataclass
from typing import Optional

@dataclass
class RandomAffine:
    degrees: float
    translate: Optional[tuple] = None
    scale: Optional[tuple] = None
    shear: Optional[tuple] = None


@dataclass
class RandomRotation:
    degrees: float


@dataclass
class HorizontalFlip:
    p: float = 0.5


@dataclass
class VerticalFlip:
    p: float = 0.5


@dataclass
class RandomResizedCrop:
    size: tuple
    scale: tuple = (0.08, 1.0)
    ratio: tuple = (0.75, 1.33)


@dataclass
class ColorJitter:
    brightness: float = 0
    contrast: float = 0
    saturation: float = 0
    hue: float = 0


class Augmentor:
    def __init__(self, augment_configs: list = None, pad: int = None):
        self.augment_configs = augment_configs or []
        self.pad = pad


    def _build_transform(self, config):
        if isinstance(config, RandomAffine):
            return transforms.RandomAffine(
                    degrees=config.degrees,
                    translate=config.translate,
                    scale=config.scale,
                    shear=config.shear
                    )
        elif isinstance(config, RandomRotation):
            return transforms.RandomRotation(config.degrees)
        elif isinstance(config, HorizontalFlip):
            return transforms.RandomHorizontalFlip(p=config.p)
        elif isinstance(config, VerticalFlip):
            return transforms.RandomVerticalFlip(p=config.p)
        elif isinstance(config, RandomResizedCrop):
            return transforms.RandomResizedCrop(config.size, scale=config.scale, ratio=config.ratio)
        elif isinstance(config, ColorJitter):
            return transforms.ColorJitter(
                    brightness=config.brightness,
                    contrast=config.contrast,
                    saturation=config.saturation,
                    hue=config.hue
                    )
        else:
            raise ValueError(f"Unrecognizes transform config type: {type(config)}")

    
    def get_train_transforms(self):
        transform_list = [self._build_transform(config) for config in self.augment_configs]
        if self.pad is not None:
            transform_list.append(transforms.Pad(self.pad))
        transform_list.append(transforms.ToTensor())

        return transforms.Compose(transform_list)


    def get_val_transforms(self):
        transform_list = []
        if self.pad is not None:
            transform_list.append(transforms.Pad(self.pad))
        transform_list.append(transforms.ToTensor())

        return transforms.Compose(transform_list)

