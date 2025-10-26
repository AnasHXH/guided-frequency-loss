"""
Additional loss components for image restoration.

This module provides various loss functions that can be used
individually or combined with the Guided Frequency Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    
    Args:
        feature_layers (list): VGG layers to use for feature extraction
        weights (list): Weights for each layer
    """
    
    def __init__(
        self,
        feature_layers: list = [3, 8, 15, 22],
        weights: list = [1.0, 1.0, 1.0, 1.0]
    ):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Extract layers
        self.slices = nn.ModuleList()
        prev_layer = 0
        for layer_idx in feature_layers:
            self.slices.append(vgg[prev_layer:layer_idx + 1])
            prev_layer = layer_idx + 1
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        loss = 0.0
        pred_features = pred
        target_features = target
        
        for i, (slice_module, weight) in enumerate(zip(self.slices, self.weights)):
            pred_features = slice_module(pred_features)
            target_features = slice_module(target_features)
            loss += weight * F.l1_loss(pred_features, target_features)
        
        return loss


class EdgeLoss(nn.Module):
    """
    Edge-aware loss using Sobel filters.
    
    Args:
        weight (float): Weight of edge loss
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute edge loss.
        
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Edge loss value
        """
        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        
        # Compute magnitude
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)
        target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)
        
        # L1 loss on gradients
        loss = F.l1_loss(pred_grad, target_grad)
        
        return self.weight * loss


class TVLoss(nn.Module):
    """
    Total Variation loss for smoothness.
    
    Args:
        weight (float): Weight of TV loss
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute TV loss.
        
        Args:
            x (torch.Tensor): Input image
            
        Returns:
            torch.Tensor: TV loss value
        """
        batch_size, c, h, w = x.size()
        
        # Horizontal and vertical differences
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss.
    
    Args:
        window_size (int): Size of Gaussian window
        max_val (float): Maximum pixel value
    """
    
    def __init__(self, window_size: int = 11, max_val: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel."""
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
        Compute SSIM loss (1 - SSIM).
        
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: SSIM loss value
        """
        window = self.window.to(pred.device)
        
        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return 1 - SSIM as loss (minimize)
        return 1 - ssim_map.mean()


class ColorLoss(nn.Module):
    """
    Color consistency loss in different color spaces.
    
    Args:
        space (str): Color space ('rgb', 'hsv', 'lab')
        weight (float): Loss weight
    """
    
    def __init__(self, space: str = 'rgb', weight: float = 1.0):
        super().__init__()
        self.space = space
        self.weight = weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute color loss.
        
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Color loss value
        """
        if self.space == 'rgb':
            loss = F.l1_loss(pred, target)
        else:
            # For other color spaces, use RGB loss as approximation
            loss = F.l1_loss(pred, target)
        
        return self.weight * loss


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GAN training.
    
    Args:
        loss_type (str): Type of GAN loss ('vanilla', 'lsgan', 'wgan')
    """
    
    def __init__(self, loss_type: str = 'vanilla'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        pred: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """
        Compute adversarial loss.
        
        Args:
            pred (torch.Tensor): Discriminator predictions
            target_is_real (bool): Whether target is real
            
        Returns:
            torch.Tensor: Adversarial loss value
        """
        if self.loss_type == 'wgan':
            return -pred.mean() if target_is_real else pred.mean()
        else:
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return self.criterion(pred, target)


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with weights.
    
    Args:
        losses (dict): Dictionary of loss functions and their weights
        
    Example:
        >>> losses = {
        >>>     'l1': (nn.L1Loss(), 1.0),
        >>>     'perceptual': (PerceptualLoss(), 0.1),
        >>>     'edge': (EdgeLoss(), 0.05)
        >>> }
        >>> combined_loss = CombinedLoss(losses)
    """
    
    def __init__(self, losses: dict):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for name, (loss_fn, weight) in losses.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_individual: bool = False
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred (torch.Tensor): Predicted image
            target (torch.Tensor): Target image
            return_individual (bool): Whether to return individual losses
            
        Returns:
            torch.Tensor or dict: Total loss or dict of individual losses
        """
        total_loss = 0.0
        individual_losses = {}
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            weighted_loss = self.weights[name] * loss_value
            total_loss += weighted_loss
            individual_losses[name] = loss_value.item()
        
        if return_individual:
            return total_loss, individual_losses
        return total_loss
