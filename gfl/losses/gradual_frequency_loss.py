"""
Gradual Frequency Loss (GFL) for Image Restoration

This module implements the Guided Frequency Loss which combines:
1. Charbonnier loss in spatial domain
2. Laplacian pyramid loss for multi-scale features
3. Gradual frequency-domain loss with progressive high-frequency emphasis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GradualFrequencyLoss(nn.Module):
    """
    Guided Frequency Loss for image restoration tasks.
    
    This loss function progressively emphasizes high-frequency components during training
    while maintaining spatial and multi-scale structural consistency.
    
    Args:
        start_frequency (int): Starting frequency for the bandpass filter. Default: 10
        end_frequency (int): Ending frequency for the bandpass filter. Default: 255
        num_epochs (int): Total number of training epochs. Default: 100
        filter_apply (bool): Whether to apply additional filtering. Default: False
        eps (float): Small constant for numerical stability. Default: 1e-3
        kernel_size (int): Gaussian kernel size for Laplacian pyramid. Default: 5
        sigma (float): Gaussian kernel sigma. Default: 1.0
    
    Example:
        >>> loss_fn = GradualFrequencyLoss(start_frequency=10, end_frequency=255, num_epochs=100)
        >>> pred = torch.randn(4, 3, 192, 192)
        >>> target = torch.randn(4, 3, 192, 192)
        >>> loss = loss_fn(pred, target, current_epoch=50)
    """
    
    def __init__(
        self,
        start_frequency: int = 10,
        end_frequency: int = 255,
        num_epochs: int = 100,
        filter_apply: bool = False,
        eps: float = 1e-3,
        kernel_size: int = 5,
        sigma: float = 1.0
    ):
        super(GradualFrequencyLoss, self).__init__()
        
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.num_epochs = num_epochs
        self.filter_apply = filter_apply
        self.eps = eps
        
        # Initialize Gaussian kernel for Laplacian pyramid
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)
        
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Create a Gaussian kernel for image filtering.
        
        Args:
            kernel_size (int): Size of the kernel
            sigma (float): Standard deviation of the Gaussian
            
        Returns:
            torch.Tensor: Gaussian kernel of shape (3, 1, kernel_size, kernel_size)
        """
        # Create coordinate grids
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        # Create 2D Gaussian
        g = coords ** 2
        g = g.view(1, -1) + g.view(-1, 1)
        g = torch.exp(-g / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Expand to 3 channels (RGB)
        kernel = g.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        
        return kernel
    
    def _calculate_frequency_range(self, current_epoch: int) -> Tuple[int, int]:
        """
        Calculate the frequency range for current epoch (gradual progression).
        
        Args:
            current_epoch (int): Current training epoch
            
        Returns:
            Tuple[int, int]: (start_freq, end_freq) for current epoch
        """
        progress = min(current_epoch / self.num_epochs, 1.0)
        
        # Gradually increase the frequency range
        current_start = self.start_frequency
        current_end = int(self.start_frequency + 
                         (self.end_frequency - self.start_frequency) * progress)
        
        return current_start, current_end
    
    def _extract_frequency_components(
        self,
        x: torch.Tensor,
        start: int,
        end: int,
        filter_apply: bool = False
    ) -> torch.Tensor:
        """
        Extract specific frequency components using FFT.
        
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W)
            start (int): Lower cutoff frequency
            end (int): Upper cutoff frequency
            filter_apply (bool): Apply additional filtering
            
        Returns:
            torch.Tensor: Filtered frequency components (B, 2, C, H, W)
                         where dim=1 contains [real, imaginary] parts
        """
        device = x.device
        b, c, h, w = x.shape
        
        # Process each channel separately
        filtered_channels = []
        
        for channel_idx in range(c):
            channel = x[:, channel_idx, :, :]  # (B, H, W)
            
            # Apply 2D FFT
            f_channel = torch.fft.fft2(channel, norm='ortho')
            
            # Create bandpass mask
            mask = torch.zeros_like(f_channel, dtype=torch.float32, device=device)
            mask[:, :, :start] = 1
            mask[:, :, end:] = 1
            
            # Apply mask
            filtered = f_channel * mask
            
            # Stack real and imaginary parts
            filtered_ri = torch.stack([filtered.real, filtered.imag], dim=0)  # (2, B, H, W)
            filtered_channels.append(filtered_ri)
        
        # Stack channels: (2, B, C, H, W) -> transpose to (B, 2, C, H, W)
        result = torch.stack(filtered_channels, dim=2).permute(1, 0, 2, 3, 4)
        
        return result
    
    def _conv_gauss(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian convolution to image.
        
        Args:
            img (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            torch.Tensor: Filtered image
        """
        n_channels, _, kw, kh = self.kernel.shape
        
        # Pad image
        img_padded = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        
        # Apply convolution
        kernel = self.kernel.to(img.device)
        filtered = F.conv2d(img_padded, kernel, groups=n_channels)
        
        return filtered
    
    def _laplacian_kernel(self, current: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian pyramid representation.
        
        Args:
            current (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            torch.Tensor: Laplacian representation
        """
        # Filter
        filtered = self._conv_gauss(current)
        
        # Downsample
        down = filtered[:, :, ::2, ::2]
        
        # Upsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        
        # Filter again
        filtered = self._conv_gauss(new_filter)
        
        # Compute difference (Laplacian)
        diff = current - filtered
        
        return diff
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        current_epoch: int
    ) -> torch.Tensor:
        """
        Compute the Guided Frequency Loss.
        
        Args:
            pred (torch.Tensor): Predicted image (B, C, H, W)
            target (torch.Tensor): Target image (B, C, H, W)
            current_epoch (int): Current training epoch
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Calculate current frequency range
        start_freq, end_freq = self._calculate_frequency_range(current_epoch)
        
        # Extract frequency components
        pred_freq = self._extract_frequency_components(
            pred, start_freq, end_freq, self.filter_apply
        )
        target_freq = self._extract_frequency_components(
            target, start_freq, end_freq, self.filter_apply
        )
        
        # Compute Laplacian pyramid differences
        laplacian_diff = self._laplacian_kernel(pred) - self._laplacian_kernel(target)
        
        # Spatial differences
        spatial_diff = pred - target
        
        # Frequency differences
        freq_diff = pred_freq - target_freq
        
        # Combined loss with Charbonnier penalty
        loss = torch.mean(
            torch.sqrt(
                (spatial_diff ** 2) +
                (freq_diff ** 2).sum(dim=1) +  # Sum over real/imaginary dimensions
                (laplacian_diff ** 2) +
                (self.eps ** 2)
            )
        )
        
        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (L1 smooth loss) for robust reconstruction.
    
    Args:
        eps (float): Small constant for numerical stability. Default: 1e-3
    """
    
    def __init__(self, eps: float = 1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Charbonnier loss.
        
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Loss value
        """
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))
        return loss


class LaplacianPyramidLoss(nn.Module):
    """
    Multi-scale Laplacian Pyramid Loss for capturing structural information.
    
    Args:
        num_levels (int): Number of pyramid levels. Default: 3
        kernel_size (int): Gaussian kernel size. Default: 5
        sigma (float): Gaussian kernel sigma. Default: 1.0
    """
    
    def __init__(
        self,
        num_levels: int = 3,
        kernel_size: int = 5,
        sigma: float = 1.0
    ):
        super(LaplacianPyramidLoss, self).__init__()
        self.num_levels = num_levels
        self.kernel = self._create_gaussian_kernel(kernel_size, sigma)
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        
        g = coords ** 2
        g = g.view(1, -1) + g.view(-1, 1)
        g = torch.exp(-g / (2 * sigma ** 2))
        g = g / g.sum()
        
        kernel = g.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        return kernel
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale Laplacian pyramid loss.
        
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Loss value
        """
        total_loss = 0.0
        
        for level in range(self.num_levels):
            # Compute Laplacian at current scale
            pred_lap = self._compute_laplacian(pred)
            target_lap = self._compute_laplacian(target)
            
            # Add to total loss
            total_loss += F.l1_loss(pred_lap, target_lap)
            
            # Downsample for next level
            if level < self.num_levels - 1:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                target = F.avg_pool2d(target, kernel_size=2, stride=2)
        
        return total_loss / self.num_levels
    
    def _compute_laplacian(self, img: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of image."""
        kernel = self.kernel.to(img.device)
        n_channels = img.shape[1]
        
        # Pad and filter
        kw = kernel.shape[-1]
        img_padded = F.pad(img, (kw//2, kw//2, kw//2, kw//2), mode='replicate')
        filtered = F.conv2d(img_padded, kernel, groups=n_channels)
        
        return img - filtered
