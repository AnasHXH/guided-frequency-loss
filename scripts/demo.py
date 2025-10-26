"""
Demo script for image super-resolution using trained GFL model.

Usage:
    python scripts/demo.py --checkpoint path/to/checkpoint.ckpt --input path/to/image.png --output path/to/output.png
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from gfl.models.swinir import SwinIR
from gfl.utils.visualization import visualize_comparison
from gfl.utils.metrics import calculate_psnr, calculate_ssim


def load_model(checkpoint_path: str, device: str = 'cuda') -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (str): Device to load model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict (handles both Lightning and pure PyTorch checkpoints)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from Lightning)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # Create model (you may need to adjust these parameters based on your checkpoint)
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=48,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path: str, upscale_factor: int = 4) -> tuple:
    """
    Preprocess image for inference.
    
    Args:
        image_path (str): Path to input image
        upscale_factor (int): Upscaling factor
        
    Returns:
        tuple: (lr_tensor, original_size)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize to create LR image
    lr_size = (original_size[0] // upscale_factor, original_size[1] // upscale_factor)
    lr_image = image.resize(lr_size, Image.BICUBIC)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    lr_tensor = transform(lr_image).unsqueeze(0)  # Add batch dimension
    
    return lr_tensor, original_size


def postprocess_image(sr_tensor: torch.Tensor, target_size: tuple = None) -> Image.Image:
    """
    Postprocess super-resolved tensor to PIL Image.
    
    Args:
        sr_tensor (torch.Tensor): Super-resolved image tensor
        target_size (tuple, optional): Target size to resize to
        
    Returns:
        Image.Image: Output image
    """
    # Remove batch dimension and move to CPU
    sr_tensor = sr_tensor.squeeze(0).cpu()
    
    # Clamp values to [0, 1]
    sr_tensor = torch.clamp(sr_tensor, 0, 1)
    
    # Convert to PIL Image
    sr_image = transforms.ToPILImage()(sr_tensor)
    
    # Resize if needed
    if target_size:
        sr_image = sr_image.resize(target_size, Image.LANCZOS)
    
    return sr_image


def super_resolve_image(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    device: str = 'cuda',
    upscale_factor: int = 4,
    visualize: bool = True
):
    """
    Perform super-resolution on an image.
    
    Args:
        model (torch.nn.Module): Trained SR model
        input_path (str): Path to input LR image
        output_path (str): Path to save SR image
        device (str): Device to use
        upscale_factor (int): Upscaling factor
        visualize (bool): Whether to visualize results
    """
    print(f"Processing image: {input_path}")
    
    # Preprocess
    lr_tensor, original_size = preprocess_image(input_path, upscale_factor)
    lr_tensor = lr_tensor.to(device)
    
    # Inference
    print("Running super-resolution...")
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    # Postprocess
    sr_image = postprocess_image(sr_tensor, original_size)
    
    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sr_image.save(output_path)
    print(f"Saved super-resolved image to: {output_path}")
    
    # Visualize if requested
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Load and display LR image
        lr_image = Image.open(input_path).convert('RGB')
        axes[0].imshow(lr_image)
        axes[0].set_title('Input (Low Resolution)')
        axes[0].axis('off')
        
        # Display SR image
        axes[1].imshow(sr_image)
        axes[1].set_title('Output (Super Resolution)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Super-resolve images using GFL model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output image')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--upscale', type=int, default=4, help='Upscale factor')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Process image
    super_resolve_image(
        model=model,
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        upscale_factor=args.upscale,
        visualize=not args.no_viz
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
