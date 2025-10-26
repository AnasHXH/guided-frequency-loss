"""
Test script for evaluating trained model on test dataset.

Usage:
    python scripts/test.py --config configs/swinir_sr_x4.yaml --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import sys
from pathlib import Path
import time

import torch
import pytorch_lightning as pl
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from gfl.losses.gradual_frequency_loss import GradualFrequencyLoss
from gfl.models.swinir import SwinIR
from gfl.data.dataset import ImageRestorationDataModule
from gfl.utils.metrics import PSNRMetric, SSIMMetric
from gfl.utils.visualization import visualize_comparison, save_image_grid


class TestModel(pl.LightningModule):
    """Simple wrapper for testing."""
    
    def __init__(self, model_config: dict):
        super().__init__()
        self.model = self._build_model(model_config)
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        
        self.test_results = []
    
    def _build_model(self, config: dict):
        model_type = config.pop('type', 'swinir')
        if model_type.lower() == 'swinir':
            return SwinIR(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        return self.model(x)
    
    def test_step(self, batch, batch_idx):
        lr_images, hr_images = batch
        
        # Measure inference time
        start_time = time.time()
        sr_images = self(lr_images)
        inference_time = time.time() - start_time
        
        # Compute metrics
        psnr = self.psnr(sr_images, hr_images)
        ssim = self.ssim(sr_images, hr_images)
        
        # Store results
        self.test_results.append({
            'psnr': psnr.item(),
            'ssim': ssim.item(),
            'inference_time': inference_time / lr_images.shape[0]  # Per image
        })
        
        # Log metrics
        self.log('test/psnr', psnr, on_step=False, on_epoch=True)
        self.log('test/ssim', ssim, on_step=False, on_epoch=True)
        
        # Save some visualizations (first batch only)
        if batch_idx == 0:
            visualize_comparison(
                lr_images=lr_images,
                sr_images=sr_images,
                hr_images=hr_images,
                num_images=min(4, lr_images.shape[0]),
                save_path='test_results/comparison.png'
            )
            save_image_grid(
                sr_images,
                save_path='test_results/sr_grid.png',
                nrow=4
            )
        
        return {'psnr': psnr, 'ssim': ssim}
    
    def on_test_epoch_end(self):
        """Print summary statistics."""
        if self.test_results:
            avg_psnr = sum(r['psnr'] for r in self.test_results) / len(self.test_results)
            avg_ssim = sum(r['ssim'] for r in self.test_results) / len(self.test_results)
            avg_time = sum(r['inference_time'] for r in self.test_results) / len(self.test_results)
            
            print(f"\n{'='*80}")
            print(f"Test Results Summary")
            print(f"{'='*80}")
            print(f"Number of images: {len(self.test_results)}")
            print(f"Average PSNR: {avg_psnr:.4f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Average inference time: {avg_time:.4f} seconds/image")
            print(f"{'='*80}\n")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test image restoration model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override batch size if specified
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # Create data module
    print("Creating data module...")
    data_module = ImageRestorationDataModule(**config['data'])
    data_module.setup('test')
    
    # Create model
    print("Creating model...")
    model = TestModel(model_config=config['model'])
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.model.load_state_dict(state_dict)
    else:
        model.model.load_state_dict(checkpoint)
    
    # Create trainer
    print("Creating trainer...")
    trainer = pl.Trainer(
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
        logger=False,
        enable_checkpointing=False
    )
    
    # Test
    print("Running test...")
    print(f"{'='*80}\n")
    trainer.test(model, data_module.test_dataloader())
    
    print("\nTest completed! Check 'test_results/' directory for visualizations.")


if __name__ == '__main__':
    main()
