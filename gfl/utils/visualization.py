"""
Visualization utilities for image restoration results.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from pathlib import Path


def denormalize(tensor: torch.Tensor, mean: List[float] = [0.5], std: List[float] = [0.5]) -> torch.Tensor:
    """
    Denormalize a normalized tensor.
    
    Args:
        tensor (torch.Tensor): Normalized tensor
        mean (list): Mean used for normalization
        std (list): Std used for normalization
        
    Returns:
        torch.Tensor: Denormalized tensor
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_comparison(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    num_images: int = 4,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Visualize comparison of LR, SR, and HR images.
    
    Args:
        lr_images (torch.Tensor): Low-resolution images (B, C, H, W)
        sr_images (torch.Tensor): Super-resolved images (B, C, H, W)
        hr_images (torch.Tensor): High-resolution images (B, C, H, W)
        num_images (int): Number of images to display
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    num_images = min(num_images, lr_images.shape[0])
    
    fig, axes = plt.subplots(num_images, 3, figsize=(figsize[0], figsize[1] * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # LR image
        lr_img = lr_images[i].cpu().permute(1, 2, 0).numpy()
        lr_img = np.clip(lr_img, 0, 1)
        axes[i, 0].imshow(lr_img)
        axes[i, 0].set_title('Low Resolution')
        axes[i, 0].axis('off')
        
        # SR image
        sr_img = sr_images[i].cpu().permute(1, 2, 0).numpy()
        sr_img = np.clip(sr_img, 0, 1)
        axes[i, 1].imshow(sr_img)
        axes[i, 1].set_title('Super Resolution')
        axes[i, 1].axis('off')
        
        # HR image
        hr_img = hr_images[i].cpu().permute(1, 2, 0).numpy()
        hr_img = np.clip(hr_img, 0, 1)
        axes[i, 2].imshow(hr_img)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_batch(
    images: torch.Tensor,
    num_images: int = 8,
    nrow: int = 4,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Visualize a batch of images in a grid.
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W)
        num_images (int): Number of images to display
        nrow (int): Number of images per row
        title (str, optional): Title for the figure
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    num_images = min(num_images, images.shape[0])
    
    # Create grid
    grid = torchvision.utils.make_grid(
        images[:num_images],
        nrow=nrow,
        normalize=True,
        value_range=(0, 1)
    )
    
    # Convert to numpy
    grid_np = grid.cpu().permute(1, 2, 0).numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid_np)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def save_image_grid(
    images: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True
):
    """
    Save a batch of images as a grid.
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W)
        save_path (str): Path to save the image
        nrow (int): Number of images per row
        normalize (bool): Whether to normalize the images
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torchvision.utils.save_image(
        images,
        save_path,
        nrow=nrow,
        normalize=normalize,
        value_range=(0, 1) if normalize else None
    )
    print(f"Saved image grid to {save_path}")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_psnr: Optional[List[float]] = None,
    val_psnr: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    Plot training curves for loss and metrics.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_psnr (list, optional): Training PSNR values
        val_psnr (list, optional): Validation PSNR values
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    num_plots = 2 if train_psnr is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot PSNR if available
    if train_psnr is not None and num_plots == 2:
        axes[1].plot(train_psnr, label='Train PSNR', linewidth=2)
        axes[1].plot(val_psnr, label='Val PSNR', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('PSNR (dB)', fontsize=12)
        axes[1].set_title('Training and Validation PSNR', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def visualize_frequency_components(
    image: torch.Tensor,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
):
    """
    Visualize image and its frequency components.
    
    Args:
        image (torch.Tensor): Image tensor (C, H, W)
        title (str, optional): Title for the figure
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    # Convert to numpy
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Convert to grayscale for FFT
    img_gray = np.mean(img_np, axis=2)
    
    # Compute FFT
    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Frequency spectrum
    axes[1].imshow(magnitude_spectrum, cmap='jet')
    axes[1].set_title('Frequency Spectrum (Log Scale)')
    axes[1].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved frequency visualization to {save_path}")
    
    plt.show()
