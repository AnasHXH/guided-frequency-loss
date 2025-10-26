"""
Custom transforms for image restoration tasks.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from typing import Tuple, Optional
from PIL import Image


class RandomCrop:
    """
    Randomly crop image to given size.
    
    Args:
        size (tuple): Output size (height, width)
    """
    
    def __init__(self, size: Tuple[int, int]):
        self.size = size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return TF.crop(
            img,
            *transforms.RandomCrop.get_params(img, self.size)
        )


class PairedRandomCrop:
    """
    Apply same random crop to paired images (LR and HR).
    
    Args:
        hr_size (tuple): Size for HR image
        scale (int): Downscaling factor
    """
    
    def __init__(self, hr_size: Tuple[int, int], scale: int = 4):
        self.hr_size = hr_size
        self.lr_size = (hr_size[0] // scale, hr_size[1] // scale)
    
    def __call__(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Get random crop parameters for HR image
        i, j, h, w = transforms.RandomCrop.get_params(hr_img, self.hr_size)
        hr_cropped = TF.crop(hr_img, i, j, h, w)
        
        # Apply corresponding crop to LR image
        scale = hr_img.size[0] // lr_img.size[0]
        lr_cropped = TF.crop(lr_img, i // scale, j // scale, h // scale, w // scale)
        
        return lr_cropped, hr_cropped


class PairedRandomHorizontalFlip:
    """
    Apply same random horizontal flip to paired images.
    
    Args:
        p (float): Probability of flip. Default: 0.5
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)
        return lr_img, hr_img


class PairedRandomVerticalFlip:
    """
    Apply same random vertical flip to paired images.
    
    Args:
        p (float): Probability of flip. Default: 0.5
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)
        return lr_img, hr_img


class PairedRandomRotation:
    """
    Apply same random 90-degree rotation to paired images.
    """
    
    def __call__(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            lr_img = TF.rotate(lr_img, angle)
            hr_img = TF.rotate(hr_img, angle)
        return lr_img, hr_img


class AddGaussianNoise:
    """
    Add Gaussian noise to image.
    
    Args:
        noise_level (float): Standard deviation of noise (0-255)
        p (float): Probability of applying noise. Default: 1.0
    """
    
    def __init__(self, noise_level: float = 25.0, p: float = 1.0):
        self.noise_level = noise_level / 255.0  # Normalize to [0, 1]
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.noise_level
            img = img + noise
            img = torch.clamp(img, 0, 1)
        return img


class RandomJPEGCompression:
    """
    Apply random JPEG compression artifacts.
    
    Args:
        quality_range (tuple): Min and max quality (1-100)
        p (float): Probability of applying compression. Default: 0.5
    """
    
    def __init__(self, quality_range: Tuple[int, int] = (50, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            from io import BytesIO
            quality = random.randint(*self.quality_range)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        return img


class DegradeLR:
    """
    Create LR image from HR using various degradation methods.
    
    Args:
        scale (int): Downscaling factor
        degradation_type (str): Type of degradation ('bicubic', 'bilinear', 'nearest')
        add_noise (bool): Whether to add noise
        noise_level (float): Noise level if add_noise is True
    """
    
    def __init__(
        self,
        scale: int = 4,
        degradation_type: str = 'bicubic',
        add_noise: bool = False,
        noise_level: float = 0.0
    ):
        self.scale = scale
        self.degradation_type = degradation_type
        self.add_noise = add_noise
        self.noise_level = noise_level
    
    def __call__(self, hr_img: Image.Image) -> Image.Image:
        # Downscale
        lr_size = (hr_img.size[0] // self.scale, hr_img.size[1] // self.scale)
        
        if self.degradation_type == 'bicubic':
            lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        elif self.degradation_type == 'bilinear':
            lr_img = hr_img.resize(lr_size, Image.BILINEAR)
        elif self.degradation_type == 'nearest':
            lr_img = hr_img.resize(lr_size, Image.NEAREST)
        else:
            lr_img = hr_img.resize(lr_size, Image.BICUBIC)
        
        # Add noise if requested
        if self.add_noise and self.noise_level > 0:
            lr_tensor = TF.to_tensor(lr_img)
            noise = torch.randn_like(lr_tensor) * (self.noise_level / 255.0)
            lr_tensor = torch.clamp(lr_tensor + noise, 0, 1)
            lr_img = TF.to_pil_image(lr_tensor)
        
        return lr_img


class ToTensorNormalize:
    """
    Convert PIL Image to tensor and normalize.
    
    Args:
        mean (tuple): Mean for normalization
        std (tuple): Std for normalization
    """
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ):
        self.mean = mean
        self.std = std
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, self.mean, self.std)
        return tensor


def get_training_augmentation(hr_size: Tuple[int, int] = (192, 192), scale: int = 4):
    """
    Get standard training augmentation pipeline for super-resolution.
    
    Args:
        hr_size (tuple): High-resolution image size
        scale (int): Upscaling factor
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(hr_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor()
    ])


def get_validation_transform(hr_size: Tuple[int, int] = (192, 192)):
    """
    Get validation transform (no augmentation).
    
    Args:
        hr_size (tuple): High-resolution image size
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(hr_size),
        transforms.ToTensor()
    ])


def get_denoising_transform(noise_level: float = 25.0):
    """
    Get transform for image denoising tasks.
    
    Args:
        noise_level (float): Standard deviation of Gaussian noise
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(noise_level=noise_level)
    ])
