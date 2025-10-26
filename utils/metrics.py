"""
Metrics for evaluating image restoration quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PSNRMetric(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) metric.
    
    Args:
        max_val (float): Maximum possible pixel value. Default: 1.0
    """
    
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate PSNR between prediction and target.
        
        Args:
            pred (torch.Tensor): Predicted image (B, C, H, W)
            target (torch.Tensor): Target image (B, C, H, W)
            
        Returns:
            torch.Tensor: PSNR value in dB
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


class SSIMMetric(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) metric.
    
    Args:
        window_size (int): Size of the Gaussian window. Default: 11
        channel (int): Number of image channels. Default: 3
        max_val (float): Maximum possible pixel value. Default: 1.0
    """
    
    def __init__(self, window_size: int = 11, channel: int = 3, max_val: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.max_val = max_val
        
        # Create Gaussian window
        self.window = self._create_window(window_size, channel)
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        gauss = torch.Tensor([
            torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate SSIM between prediction and target.
        
        Args:
            pred (torch.Tensor): Predicted image (B, C, H, W)
            target (torch.Tensor): Target image (B, C, H, W)
            
        Returns:
            torch.Tensor: SSIM value (range: 0-1, higher is better)
        """
        # Move window to same device as input
        window = self.window.to(pred.device)
        
        # Constants for stability
        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        
        # Compute means
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate PSNR between two images.
    
    Args:
        pred (torch.Tensor): Predicted image
        target (torch.Tensor): Target image
        max_val (float): Maximum pixel value
        
    Returns:
        float: PSNR value in dB
    """
    metric = PSNRMetric(max_val=max_val)
    return metric(pred, target).item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate SSIM between two images.
    
    Args:
        pred (torch.Tensor): Predicted image
        target (torch.Tensor): Target image
        max_val (float): Maximum pixel value
        
    Returns:
        float: SSIM value
    """
    metric = SSIMMetric(max_val=max_val)
    return metric(pred, target).item()
