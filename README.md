# Guided Frequency Loss (GFL) for Image Restoration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **Guided Frequency Loss (GFL)**, a novel loss function for image restoration tasks including Super-Resolution and Denoising.

## 📖 Abstract

Recent advancements in multimedia image enhancement and restoration have leveraged various generative models to achieve significant improvements. Despite this progress, the potential of utilizing the frequency domain remains underexplored, which is crucial for effectively addressing complex scenarios. 

This repository introduces the **Guided Frequency Loss (GFL)**, a novel approach designed to optimize the learning of frequency content in balance with the spatial patterns in images. It aggregates three major components that work in parallel to enhance learning efficiency:
- **Charbonnier component**
- **Laplacian Pyramid component**
- **Gradual Frequency component**

## ✨ Key Features

- 🎯 **Joint Learning**: Simultaneously learns spatial structures and frequency details for image restoration
- 🚀 **Enhanced Training**: Improves training stability and convergence across diverse deep models (Transformer & GAN-based)
- 📈 **Progressive Strategy**: Utilizes Gradual Frequency approach to progressively capture complex high-frequency components
- 🎨 **Better Reconstruction**: Strengthens texture and edge reconstruction through integrated loss terms
- 💪 **Data Efficient**: Improves restoration accuracy (PSNR/SSIM) even with limited training data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/guided-frequency-loss.git
cd guided-frequency-loss

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from gfl.losses import GradualFrequencyLoss
from gfl.models import SwinIRSuperResolution
import torch

# Initialize the loss function
gfl_loss = GradualFrequencyLoss(
    start_frequency=10,
    end_frequency=255,
    num_epochs=100,
    filter_apply=False
)

# Create model
model = SwinIRSuperResolution(upscale=4, window_size=8)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        lr_images, hr_images = batch
        
        # Forward pass
        sr_images = model(lr_images)
        
        # Compute GFL loss
        loss = gfl_loss(sr_images, hr_images, current_epoch=epoch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

## 📂 Project Structure

```
guided-frequency-loss/
├── gfl/
│   ├── __init__.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── gradual_frequency_loss.py
│   │   └── components.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── swinir.py
│   │   └── layers.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
├── configs/
│   ├── swinir_sr_x4.yaml
│   └── training_config.yaml
├── scripts/
│   ├── train.py
│   ├── test.py
│   └── demo.py
├── notebooks/
│   └── demo_tutorial.ipynb
├── tests/
│   └── test_loss.py
├── requirements.txt
├── setup.py
└── README.md
```

## 🎓 Training

### Super-Resolution (4x)

```bash
python scripts/train.py \
    --task super_resolution \
    --upscale 4 \
    --batch_size 8 \
    --num_epochs 100 \
    --data_path /path/to/dataset \
    --model swinir
```

### Image Denoising

```bash
python scripts/train.py \
    --task denoising \
    --noise_level 25 \
    --batch_size 8 \
    --num_epochs 100 \
    --data_path /path/to/dataset
```

## 📊 Results

### Super-Resolution Performance

| Model | Dataset | PSNR (dB) | SSIM |
|-------|---------|-----------|------|
| SwinIR + GFL | DIV2K | 24.54 | 0.761 |
| SwinIR Baseline | DIV2K | 23.12 | 0.742 |
| SRGAN + GFL | BSD100 | 25.89 | 0.783 |

### Training Stability

GFL significantly improves training effectiveness by:
- Reducing stochastic variations in high-frequency components
- Achieving faster convergence
- Producing more consistent and high-quality image restoration

## 🔬 Technical Details

### Guided Frequency Loss Components

1. **Charbonnier Loss**: Provides robust reconstruction in spatial domain
2. **Laplacian Pyramid Loss**: Captures multi-scale structural information
3. **Gradual Frequency Loss**: Progressively emphasizes frequency components during training

The loss is computed as:

```
GFL = sqrt((spatial_diff)² + (spatial_charbonnier)² + (laplacian_diff)² + ε²)
```

with frequency-domain filtering applied progressively based on training epoch.

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{gfl2024,
  title={Guided Frequency Loss for Image Restoration},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 🛠️ Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- PyTorch Lightning >= 1.5.0
- torchvision >= 0.10.0
- numpy
- PIL
- matplotlib
- tqdm

See `requirements.txt` for full dependencies.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SwinIR architecture based on [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)
- Inspired by frequency-domain analysis techniques in image processing

## 📧 Contact

For questions and discussions, please open an issue or contact [your.email@example.com]

## 🔗 Links

- [Paper](link-to-paper)
- [Project Page](link-to-project-page)
- [Documentation](link-to-docs)
