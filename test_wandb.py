#!/usr/bin/env python3
"""
Test script for wandb integration with BitNet-QDyT-v2 training.
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import Trainer

def test_wandb_integration():
    """Test wandb integration with minimal arguments."""
    
    # Create minimal arguments for testing
    args = argparse.Namespace(
        model_size='small',  # Use small model for quick test
        dataset='wikitext-2-v1',  # Use smaller dataset
        tokenizer='bert-base-uncased',
        batch_size=2,  # Very small batch for testing
        learning_rate=1e-4,
        num_epochs=1,
        warmup_steps=10,
        max_length=128,  # Short sequences
        fp16=False,
        grad_clip=1.0,
        weight_decay=0.01,
        scheduler='cosine',
        mlm_probability=0.15,
        gradient_accumulation_steps=1,
        num_workers=1,
        output_dir='./test_outputs',
        log_interval=5,  # Log frequently for testing
        eval_interval=20,
        save_interval=50,
        resume_from=None,
        cache_dir='./test_cache',
        # Wandb arguments
        wandb_project='bitnet-qdyt-test',
        wandb_run_name='test-wandb-integration',
        wandb_tags=['test', 'integration'],
        wandb_notes='Testing wandb integration with BitNet-QDyT-v2',
        disable_wandb=False,
        wandb_mode='offline'  # Use offline mode for testing
    )
    
    print("Testing wandb integration...")
    print(f"Project: {args.wandb_project}")
    print(f"Run name: {args.wandb_run_name}")
    print(f"Tags: {args.wandb_tags}")
    
    try:
        # Initialize trainer (this will test wandb setup)
        trainer = Trainer(args)
        print("✓ Trainer initialized successfully")
        print("✓ Wandb integration test completed")
        
        # Clean up
        if trainer.wandb_run:
            trainer.wandb_run.finish()
            print("✓ Wandb run finished")
            
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_wandb_integration()
    sys.exit(0 if success else 1)