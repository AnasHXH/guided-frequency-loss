"""
Training script for image restoration using Guided Frequency Loss.

Usage:
    python scripts/train.py --config configs/swinir_sr_x4.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gfl.losses.gradual_frequency_loss import GradualFrequencyLoss
from gfl.models.swinir import SwinIR
from gfl.data.dataset import ImageRestorationDataModule
from gfl.utils.metrics import PSNRMetric, SSIMMetric


class ImageRestorationModel(pl.LightningModule):
    """
    PyTorch Lightning module for image restoration.
    
    Args:
        model_config (dict): Configuration for the restoration model
        loss_config (dict): Configuration for the loss function
        optimizer_config (dict): Configuration for the optimizer
        scheduler_config (dict): Configuration for the learning rate scheduler
    """
    
    def __init__(
        self,
        model_config: dict,
        loss_config: dict,
        optimizer_config: dict,
        scheduler_config: dict
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = self._build_model(model_config)
        
        # Initialize loss function
        self.criterion = GradualFrequencyLoss(**loss_config)
        
        # Initialize metrics
        self.psnr = PSNRMetric()
        self.ssim = SSIMMetric()
        
        # Store configs
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        # Lists to track metrics
        self.train_psnr_list = []
        self.val_psnr_list = []
    
    def _build_model(self, config: dict) -> torch.nn.Module:
        """Build the image restoration model."""
        model_type = config.pop('type', 'swinir')
        
        if model_type.lower() == 'swinir':
            return SwinIR(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        lr_images, hr_images = batch
        
        # Forward pass
        sr_images = self(lr_images)
        
        # Compute loss
        loss = self.criterion(sr_images, hr_images, self.current_epoch)
        
        # Compute metrics
        psnr = self.psnr(sr_images, hr_images)
        ssim = self.ssim(sr_images, hr_images)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/ssim', ssim, on_step=True, on_epoch=True)
        
        self.train_psnr_list.append(psnr.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        lr_images, hr_images = batch
        
        # Forward pass
        sr_images = self(lr_images)
        
        # Compute loss
        loss = self.criterion(sr_images, hr_images, self.current_epoch)
        
        # Compute metrics
        psnr = self.psnr(sr_images, hr_images)
        ssim = self.ssim(sr_images, hr_images)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/psnr', psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/ssim', ssim, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_psnr_list.append(psnr.item())
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        lr_images, hr_images = batch
        
        # Forward pass
        sr_images = self(lr_images)
        
        # Compute loss
        loss = self.criterion(sr_images, hr_images, self.current_epoch)
        
        # Compute metrics
        psnr = self.psnr(sr_images, hr_images)
        ssim = self.ssim(sr_images, hr_images)
        
        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/psnr', psnr, on_step=False, on_epoch=True)
        self.log('test/ssim', ssim, on_step=False, on_epoch=True)
        
        return {'test_loss': loss, 'test_psnr': psnr, 'test_ssim': ssim}
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.train_psnr_list:
            avg_psnr = sum(self.train_psnr_list) / len(self.train_psnr_list)
            print(f"\n{'='*80}")
            print(f"Training Results - Epoch {self.current_epoch}")
            print(f"Average Train PSNR: {avg_psnr:.4f} dB")
            print(f"{'='*80}\n")
            self.train_psnr_list = []
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_psnr_list:
            avg_psnr = sum(self.val_psnr_list) / len(self.val_psnr_list)
            print(f"\n{'='*80}")
            print(f"Validation Results - Epoch {self.current_epoch}")
            print(f"Average Val PSNR: {avg_psnr:.4f} dB")
            print(f"{'='*80}\n")
            self.val_psnr_list = []
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Optimizer
        optimizer_type = self.optimizer_config.get('type', 'adam').lower()
        lr = self.optimizer_config.get('lr', 2e-4)
        weight_decay = self.optimizer_config.get('weight_decay', 0)
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Learning rate scheduler
        if self.scheduler_config:
            scheduler_type = self.scheduler_config.get('type', 'reduce_on_plateau').lower()
            
            if scheduler_type == 'reduce_on_plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=self.scheduler_config.get('factor', 0.2),
                    patience=self.scheduler_config.get('patience', 5),
                    min_lr=self.scheduler_config.get('min_lr', 5e-5)
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val/loss',
                        'frequency': 1
                    }
                }
            elif scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.scheduler_config.get('T_max', 100),
                    eta_min=self.scheduler_config.get('eta_min', 1e-6)
                )
                return [optimizer], [scheduler]
            elif scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.scheduler_config.get('step_size', 30),
                    gamma=self.scheduler_config.get('gamma', 0.1)
                )
                return [optimizer], [scheduler]
        
        return optimizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train image restoration model with GFL')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--strategy', type=str, default='auto', help='Training strategy')
    parser.add_argument('--precision', type=str, default='32', help='Training precision')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set random seed
    pl.seed_everything(config.get('seed', 42))
    
    # Create data module
    print("Creating data module...")
    data_module = ImageRestorationDataModule(**config['data'])
    
    # Create model
    print("Creating model...")
    model = ImageRestorationModel(
        model_config=config['model'],
        loss_config=config['loss'],
        optimizer_config=config['optimizer'],
        scheduler_config=config.get('scheduler', {})
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training'].get('checkpoint_dir', 'checkpoints'),
        filename='{epoch}-{val/psnr:.2f}',
        monitor='val/psnr',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    early_stop_callback = EarlyStopping(
        monitor='val/psnr',
        patience=config['training'].get('early_stopping_patience', 20),
        mode='max',
        verbose=True
    )
    
    callbacks = [checkpoint_callback, lr_monitor, early_stop_callback]
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=config['training'].get('log_dir', 'logs'),
        name=config['training'].get('experiment_name', 'gfl_experiment')
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
        strategy=args.strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['training'].get('log_every_n_steps', 50),
        val_check_interval=config['training'].get('val_check_interval', 1.0),
        gradient_clip_val=config['training'].get('gradient_clip_val', 0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1)
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, data_module, ckpt_path=args.resume)
    
    # Test
    print("Running final test...")
    trainer.test(model, data_module)
    
    print("Training completed!")


if __name__ == '__main__':
    main()
