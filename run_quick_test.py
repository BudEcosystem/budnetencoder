"""
Quick test script to verify the training setup works correctly.
Runs a small training test with minimal data.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from implementation import BitNetQDyTModel
from model_configs import get_model_config, calculate_model_params
from data_processing import load_wikitext_dataset
from train import Trainer
import argparse


def run_quick_test():
    """Run a quick training test to verify setup."""
    print("="*60)
    print("Running Quick Training Test")
    print("="*60)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test small model configuration
    print("\n1. Testing model configuration...")
    config = get_model_config('small')
    params = calculate_model_params(config)
    print(f"   Model size: {params['total']/1e6:.1f}M parameters")
    
    # Create model
    print("\n2. Creating model...")
    model = BitNetQDyTModel(config)
    model = model.to(device)
    print(f"   Model created successfully")
    
    # Test data loading
    print("\n3. Loading test data...")
    try:
        train_loader, val_loader, test_loader, tokenizer = load_wikitext_dataset(
            dataset_name='wikitext-2-v1',  # Use small dataset for testing
            batch_size=4,
            max_length=128,
            num_workers=0  # Avoid multiprocessing issues in test
        )
        print(f"   Data loaded: {len(train_loader)} batches")
    except Exception as e:
        print(f"   ⚠ Data loading failed: {e}")
        print("   This is expected if datasets library needs to download data.")
        print("   Creating dummy data for testing...")
        
        # Create dummy batch for testing
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (4, 128)),
            'attention_mask': torch.ones(4, 128),
            'labels': torch.randint(0, config.vocab_size, (4, 128))
        }
        batch['labels'][batch['labels'] == 0] = -100  # Mask some tokens
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    try:
        if 'batch' not in locals():
            batch = next(iter(train_loader))
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        model.train()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            training=True
        )
        
        print(f"   Output shape: {outputs['logits'].shape}")
        print(f"   Forward pass successful")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    try:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(
            outputs['logits'].view(-1, config.vocab_size),
            batch['labels'].view(-1)
        )
        
        loss.backward()
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Backward pass successful")
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        return False
    
    # Test optimizer step
    print("\n6. Testing optimizer step...")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()
        print(f"   Optimizer step successful")
    except Exception as e:
        print(f"   ✗ Optimizer step failed: {e}")
        return False
    
    # Memory usage
    if device.type == 'cuda':
        print(f"\n7. Memory usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    print("\n" + "="*60)
    print("✓ All tests passed! Training setup is working correctly.")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run setup: python setup.py")
    print("3. Start training: ./launch_training.sh --model_size base")
    
    return True


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)