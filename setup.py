"""
Setup script for Guided Frequency Loss package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="guided-frequency-loss",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Guided Frequency Loss for Image Restoration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnasHXH/guided-frequency-loss",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pytorch-lightning>=1.5.0",
        "timm>=0.4.12",
        "Pillow>=8.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "torchmetrics>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    keywords="image-restoration super-resolution denoising deep-learning pytorch",
    project_urls={
        "Bug Reports": "https://github.com/AnasHXH/guided-frequency-loss/issues",
        "Source": "https://github.com/AnasHXH/guided-frequency-loss",
    },
)
