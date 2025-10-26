"""
Dataset and DataModule for image restoration tasks.
"""

import glob
import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from PIL import Image
import torchvision.transforms as transforms


class ImageRestorationDataset(Dataset):
    """
    Dataset for image restoration tasks (Super-Resolution, Denoising, etc.).
    
    Args:
        image_dir (str): Directory containing images
        hr_size (tuple): Size of high-resolution images (height, width)
        upscale_factor (int): Upscaling factor for super-resolution. Default: 4
        transform (callable, optional): Transform to apply to images
        degradation_type (str): Type of degradation ('bicubic', 'noise', etc.). Default: 'bicubic'
    """
    
    def __init__(
        self,
        image_dir: str,
        hr_size: Tuple[int, int] = (192, 192),
        upscale_factor: int = 4,
        transform: Optional[Callable] = None,
        degradation_type: str = 'bicubic'
    ):
        super().__init__()
        self.image_dir = image_dir
        self.hr_size = hr_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.degradation_type = degradation_type
        
        # Get all image paths
        self.image_paths = self._get_image_paths(image_dir)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def _get_image_paths(self, image_dir: str) -> List[str]:
        """Get all image paths from directory."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        return sorted(image_paths)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns:
            tuple: (lr_image, hr_image) where lr_image is low-resolution and hr_image is high-resolution
        """
        # Load image
        image_path = self.image_paths[idx]
        hr_image = Image.open(image_path).convert('RGB')
        
        # Apply transform to get HR image
        if self.transform:
            hr_image = self.transform(hr_image)
        else:
            # Default transform
            hr_image = transforms.Compose([
                transforms.Resize(self.hr_size),
                transforms.ToTensor()
            ])(hr_image)
        
        # Create LR image by downsampling
        lr_size = (self.hr_size[0] // self.upscale_factor, 
                   self.hr_size[1] // self.upscale_factor)
        
        if self.degradation_type == 'bicubic':
            lr_image = transforms.Resize(lr_size, 
                                         interpolation=transforms.InterpolationMode.BICUBIC)(hr_image)
        else:
            lr_image = transforms.Resize(lr_size)(hr_image)
        
        return lr_image, hr_image


class ImageRestorationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for image restoration.
    
    Args:
        train_dir (str): Training data directory
        val_dir (str, optional): Validation data directory
        test_dir (str, optional): Test data directory
        hr_size (tuple): Size of high-resolution images
        upscale_factor (int): Upscaling factor
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        train_split (float): Train/val split ratio if val_dir not provided
        augmentation (bool): Whether to use data augmentation
    """
    
    def __init__(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        hr_size: Tuple[int, int] = (192, 192),
        upscale_factor: int = 4,
        batch_size: int = 8,
        num_workers: int = 4,
        train_split: float = 0.9,
        augmentation: bool = True,
        **kwargs
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.hr_size = hr_size
        self.upscale_factor = upscale_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.augmentation = augmentation
        
        # Define transforms
        self.train_transform = self._get_train_transform() if augmentation else self._get_base_transform()
        self.val_transform = self._get_base_transform()
        self.test_transform = self._get_base_transform()
    
    def _get_base_transform(self) -> transforms.Compose:
        """Get base transform without augmentation."""
        return transforms.Compose([
            transforms.Resize(self.hr_size),
            transforms.ToTensor()
        ])
    
    def _get_train_transform(self) -> transforms.Compose:
        """Get training transform with augmentation."""
        return transforms.Compose([
            transforms.Resize(self.hr_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor()
        ])
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == 'fit' or stage is None:
            # Create full training dataset
            full_dataset = ImageRestorationDataset(
                image_dir=self.train_dir,
                hr_size=self.hr_size,
                upscale_factor=self.upscale_factor,
                transform=self.train_transform
            )
            
            # Split into train and validation if val_dir not provided
            if self.val_dir is None:
                train_size = int(self.train_split * len(full_dataset))
                val_size = len(full_dataset) - train_size
                self.train_dataset, self.val_dataset = random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42)
                )
            else:
                self.train_dataset = full_dataset
                self.val_dataset = ImageRestorationDataset(
                    image_dir=self.val_dir,
                    hr_size=self.hr_size,
                    upscale_factor=self.upscale_factor,
                    transform=self.val_transform
                )
        
        if stage == 'test' or stage is None:
            if self.test_dir:
                self.test_dataset = ImageRestorationDataset(
                    image_dir=self.test_dir,
                    hr_size=self.hr_size,
                    upscale_factor=self.upscale_factor,
                    transform=self.test_transform
                )
            else:
                # Use validation dataset for testing if test_dir not provided
                self.test_dataset = self.val_dataset
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class DenoisingDataset(Dataset):
    """
    Dataset for image denoising tasks.
    
    Args:
        image_dir (str): Directory containing clean images
        image_size (tuple): Size of images
        noise_level (float): Standard deviation of Gaussian noise (0-255)
        transform (callable, optional): Transform to apply to images
    """
    
    def __init__(
        self,
        image_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        noise_level: float = 25.0,
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.image_dir = image_dir
        self.image_size = image_size
        self.noise_level = noise_level / 255.0  # Normalize to [0, 1]
        self.transform = transform
        
        # Get all image paths
        self.image_paths = self._get_image_paths(image_dir)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def _get_image_paths(self, image_dir: str) -> List[str]:
        """Get all image paths from directory."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        return sorted(image_paths)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns:
            tuple: (noisy_image, clean_image)
        """
        # Load image
        image_path = self.image_paths[idx]
        clean_image = Image.open(image_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            clean_image = self.transform(clean_image)
        else:
            clean_image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])(clean_image)
        
        # Add Gaussian noise
        noise = torch.randn_like(clean_image) * self.noise_level
        noisy_image = clean_image + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)
        
        return noisy_image, clean_image
