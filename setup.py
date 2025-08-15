"""
Setup script for BitNet-QDyT-v2 training environment.
Installs dependencies and prepares the environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command with error handling."""
    print(f"\n{description}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False
    return True


def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available - training will be slow on CPU")
            return False
    except ImportError:
        print("✗ PyTorch not installed yet")
        return False


def setup_environment():
    """Setup the training environment."""
    print("="*60)
    print("BitNet-QDyT-v2 Training Environment Setup")
    print("="*60)
    
    # Create necessary directories
    directories = ['cache', 'outputs', 'outputs/checkpoints', 'outputs/logs']
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directories: {', '.join(directories)}")
    
    # Install dependencies
    if not run_command(
        "pip install -r requirements.txt",
        "Installing dependencies"
    ):
        print("\n⚠ Failed to install dependencies. Please install manually:")
        print("  pip install -r requirements.txt")
        return False
    
    # Check CUDA
    print("\nChecking CUDA availability...")
    cuda_available = check_cuda()
    
    # Download tokenizer if not cached
    print("\nPreparing tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')
        print("✓ Tokenizer ready")
    except Exception as e:
        print(f"⚠ Failed to download tokenizer: {e}")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    
    # Print usage instructions
    print("\nTo start training:")
    print("\n1. For single GPU:")
    print("   python train.py --model_size base --dataset wikitext-103-v1")
    
    print("\n2. For multiple GPUs:")
    print("   ./launch_training.sh --model_size base --dataset wikitext-103-v1")
    
    print("\n3. With mixed precision (faster):")
    print("   ./launch_training.sh --model_size base --dataset wikitext-103-v1 --fp16")
    
    print("\n4. To resume training:")
    print("   ./launch_training.sh --resume outputs/checkpoints/best.pt")
    
    print("\nFor help:")
    print("   ./launch_training.sh --help")
    
    if not cuda_available:
        print("\n⚠ WARNING: CUDA not available. Training will be very slow.")
        print("  Consider using a machine with GPU support.")
    
    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)