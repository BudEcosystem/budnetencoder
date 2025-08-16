"""
Production training script for BitNet-QDyT-v2 model.
Supports distributed training, mixed precision, checkpointing, and logging.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import wandb
import psutil
import GPUtil

# Import our modules
from implementation import BitNetQDyTModel, ProgressiveQuantizationScheduler, QDropContext, calibrate_qdyt_norms
from model_configs import get_model_config, calculate_model_params
from data_processing import load_wikitext_dataset
from evaluate import evaluate_model, calculate_perplexity

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Production trainer for BitNet-QDyT-v2 model."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup distributed training
        self.setup_distributed()
        
        # Initialize Weights & Biases
        self.setup_wandb()
        
        # Setup directories
        self.setup_directories()
        
        # Load configuration
        self.config = get_model_config(args.model_size)
        self.update_config_from_args()
        
        # Calculate and log model parameters
        param_info = calculate_model_params(self.config)
        logger.info(f"Model size: {args.model_size}")
        logger.info(f"Total parameters: {param_info['total']:,} ({param_info['total']/1e6:.1f}M)")
        
        # Load data
        self.load_data()
        
        # Initialize model
        self.setup_model()
        
        # Setup optimization
        self.setup_optimization()
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Load checkpoint if resuming
        if args.resume_from:
            self.load_checkpoint(args.resume_from)
    
    def setup_distributed(self):
        """Setup distributed training if available."""
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        
        if 'WORLD_SIZE' in os.environ:
            self.distributed = True
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            
            logger.info(f"Distributed training: rank {self.rank}/{self.world_size}")
    
    def setup_wandb(self):
        """Initialize Weights & Biases tracking."""
        if self.rank == 0 and not getattr(self.args, 'disable_wandb', False):  # Only initialize on main process
            try:
                # Create run name with timestamp and model size
                run_name = self.args.wandb_run_name or f"bitnet-qdyt-{self.args.model_size}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Try to initialize wandb with offline mode as fallback
                mode = getattr(self.args, 'wandb_mode', 'online')
                
                wandb.init(
                    project=self.args.wandb_project,
                    name=run_name,
                    tags=self.args.wandb_tags + [self.args.model_size, self.args.dataset],
                    notes=self.args.wandb_notes,
                    mode=mode,  # Can be 'online', 'offline', or 'disabled'
                    config={
                        "model_size": self.args.model_size,
                        "dataset": self.args.dataset,
                        "batch_size": self.args.batch_size,
                        "learning_rate": self.args.learning_rate,
                        "num_epochs": self.args.num_epochs,
                        "warmup_steps": self.args.warmup_steps,
                        "max_length": self.args.max_length,
                        "fp16": self.args.fp16,
                        "grad_clip": self.args.grad_clip,
                        "weight_decay": self.args.weight_decay,
                        "scheduler": self.args.scheduler,
                        "mlm_probability": self.args.mlm_probability,
                        "gradient_accumulation_steps": getattr(self.args, 'gradient_accumulation_steps', 1),
                        "device": str(self.device),
                        "world_size": self.world_size,
                        "tokenizer": self.args.tokenizer,
                        "log_interval": self.args.log_interval,
                        "eval_interval": self.args.eval_interval,
                        "save_interval": self.args.save_interval,
                    }
                )
                self.wandb_run = wandb
                logger.info(f"Initialized wandb run: {run_name} (mode: {mode})")
                
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                logger.info("Falling back to offline mode...")
                try:
                    wandb.init(
                        project=self.args.wandb_project,
                        name=run_name,
                        mode='offline',
                        config={
                            "model_size": self.args.model_size,
                            "dataset": self.args.dataset,
                            "batch_size": self.args.batch_size,
                            "learning_rate": self.args.learning_rate,
                        }
                    )
                    self.wandb_run = wandb
                    logger.info(f"Initialized wandb run in offline mode: {run_name}")
                except Exception as e2:
                    logger.warning(f"Failed to initialize wandb in offline mode: {e2}")
                    logger.info("Continuing without wandb logging")
                    self.wandb_run = None
        else:
            self.wandb_run = None
            if self.rank == 0:
                logger.info("Wandb disabled or not main process")
    
    def setup_directories(self):
        """Create necessary directories."""
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
    
    def update_config_from_args(self):
        """Update config with command line arguments."""
        if self.args.batch_size:
            self.config.batch_size = self.args.batch_size
        if self.args.learning_rate:
            self.config.learning_rate = self.args.learning_rate
        if self.args.max_length:
            self.config.max_position_embeddings = self.args.max_length
    
    def load_data(self):
        """Load WikiText dataset."""
        logger.info(f"Loading {self.args.dataset} dataset...")
        
        self.train_loader, self.val_loader, self.test_loader, self.tokenizer = load_wikitext_dataset(
            dataset_name=self.args.dataset,
            tokenizer_name=self.args.tokenizer,
            max_length=self.args.max_length,
            mlm_probability=self.args.mlm_probability,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            cache_dir=self.args.cache_dir
        )
        
        # Adjust steps based on data
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * self.args.num_epochs
        
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        logger.info(f"Total training steps: {self.total_steps}")
    
    def setup_model(self):
        """Initialize model and move to device."""
        logger.info("Initializing model...")
        
        # Update config with tokenizer vocab size
        self.config.vocab_size = self.tokenizer.vocab_size
        
        # Create model
        self.model = BitNetQDyTModel(self.config)
        self.model = self.model.to(self.device)
        
        # Setup distributed model
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Log model architecture to wandb
        if self.rank == 0 and self.wandb_run:
            # Update wandb config with model info
            self.wandb_run.config.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'vocab_size': self.config.vocab_size,
                'hidden_size': self.config.hidden_size,
                'num_hidden_layers': self.config.num_hidden_layers,
                'num_attention_heads': self.config.num_attention_heads,
                'intermediate_size': self.config.intermediate_size,
                'max_position_embeddings': self.config.max_position_embeddings,
                'steps_per_epoch': self.steps_per_epoch,
                'total_training_steps': self.total_steps,
            })
            
            # Watch model for gradient tracking (optional, can be memory intensive)
            # self.wandb_run.watch(self.model, log='all', log_freq=self.args.log_interval)
    
    def setup_optimization(self):
        """Setup optimizer, scheduler, and training utilities."""
        # Progressive quantization scheduler
        self.prog_scheduler = ProgressiveQuantizationScheduler(self.config, self.total_steps)
        
        # Collect parameters into non-overlapping groups
        quantizer_scale_params = []
        quantizer_threshold_params = []
        norm_params = []
        regular_params = []
        
        for name, param in self.model.named_parameters():
            if 'quantizer' in name and 's_tilde' in name:
                quantizer_scale_params.append(param)
            elif 'quantizer' in name and ('t' == name.split('.')[-1] or 'delta' in name):
                quantizer_threshold_params.append(param)
            elif 'qdyt' in name.lower() or 'norm' in name:
                norm_params.append(param)
            else:
                regular_params.append(param)
        
        # Create optimizer groups (only include non-empty groups)
        optimizer_groups = []
        
        if quantizer_scale_params:
            optimizer_groups.append({
                'params': quantizer_scale_params,
                'lr': self.args.learning_rate * 0.1,
                'weight_decay': 0
            })
        
        if quantizer_threshold_params:
            optimizer_groups.append({
                'params': quantizer_threshold_params,
                'lr': self.args.learning_rate * 0.1,
                'weight_decay': 0
            })
        
        if norm_params:
            optimizer_groups.append({
                'params': norm_params,
                'lr': self.args.learning_rate * 0.5,
                'weight_decay': 0
            })
        
        if regular_params:
            optimizer_groups.append({
                'params': regular_params,
                'lr': self.args.learning_rate,
                'weight_decay': self.args.weight_decay
            })
        
        # Fallback to simple optimizer if no groups
        if not optimizer_groups:
            optimizer_groups = [{'params': self.model.parameters()}]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(0.9, 0.98),
            eps=1e-8
        )
        
        # Learning rate scheduler
        if self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.total_steps
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.total_steps
            )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.args.fp16 else None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def get_system_metrics(self):
        """Get system and GPU metrics for monitoring."""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent()
        metrics['memory_percent'] = psutil.virtual_memory().percent
        
        # GPU metrics
        if torch.cuda.is_available():
            try:
                gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
                if gpu:
                    metrics['gpu_utilization'] = gpu.load * 100
                    metrics['gpu_memory_used'] = gpu.memoryUsed
                    metrics['gpu_memory_total'] = gpu.memoryTotal
                    metrics['gpu_memory_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    metrics['gpu_temperature'] = gpu.temperature
                
                # PyTorch GPU memory
                if torch.cuda.is_available():
                    metrics['torch_gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                    metrics['torch_gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
            except Exception as e:
                logger.warning(f"Could not get GPU metrics: {e}")
        
        return metrics
    
    def log_metrics(self, metrics_dict, step, prefix=""):
        """Log metrics to both wandb and tensorboard."""
        if self.rank == 0:
            # Add prefix to metric names
            prefixed_metrics = {f"{prefix}{k}" if prefix else k: v for k, v in metrics_dict.items()}
            
            # Log to wandb
            if self.wandb_run:
                self.wandb_run.log(prefixed_metrics, step=step)
            
            # Log to tensorboard
            if self.writer:
                for key, value in prefixed_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(key, value, step)
    
    def setup_logging(self):
        """Setup TensorBoard and logging."""
        if self.rank == 0:
            self.writer = SummaryWriter(self.log_dir / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        else:
            self.writer = None
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_steps = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            disable=self.rank != 0
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get current training stage
            stage = self.prog_scheduler.get_stage(self.global_step)
            qdrop_prob = self.prog_scheduler.get_qdrop_prob(self.global_step)
            
            # Forward pass with mixed precision
            with QDropContext(self.model, qdrop_prob):
                if self.args.fp16:
                    with autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            training=True
                        )
                        logits = outputs['logits']
                        loss = self.criterion(
                            logits.view(-1, self.config.vocab_size),
                            batch['labels'].view(-1)
                        )
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        training=True
                    )
                    logits = outputs['logits']
                    loss = self.criterion(
                        logits.view(-1, self.config.vocab_size),
                        batch['labels'].view(-1)
                    )
            
            # Scale loss for gradient accumulation
            if hasattr(self.args, 'gradient_accumulation_steps'):
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            if self.args.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step only on accumulation boundaries
            if hasattr(self.args, 'gradient_accumulation_steps'):
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.args.fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    
                    # LSQ gradient clipping for quantizers
                    self.clip_quantizer_gradients()
                    
                    # Optimizer step
                    if self.args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                # Original behavior without gradient accumulation
                if self.args.fp16:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.clip_quantizer_gradients()
                if self.args.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_steps += 1
            self.global_step += 1
            
            # Update progress bar
            if self.rank == 0:
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0],
                    'stage': stage,
                    'qdrop': f'{qdrop_prob:.3f}'
                })
            
            # Logging
            if self.global_step % self.args.log_interval == 0:
                # Get system metrics
                system_metrics = self.get_system_metrics()
                
                # Prepare training metrics
                train_metrics = {
                    'train/loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/qdrop_probability': qdrop_prob,
                    'train/training_stage': stage,
                    'train/epoch': self.epoch,
                    'train/global_step': self.global_step,
                }
                
                # Add gradient norms
                if hasattr(self, 'model'):
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    train_metrics['train/grad_norm'] = total_norm ** (1. / 2)
                
                # Combine all metrics
                all_metrics = {**train_metrics, **system_metrics}
                
                # Log to wandb and tensorboard
                self.log_metrics(all_metrics, self.global_step)
            
            # Validation
            if self.global_step % self.args.eval_interval == 0:
                val_metrics = self.validate()
                self.model.train()
                
                if self.rank == 0:
                    # Save checkpoint if best
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint('best')
            
            # Regular checkpoint
            if self.global_step % self.args.save_interval == 0 and self.rank == 0:
                self.save_checkpoint(f'step_{self.global_step}')
            
            # Reduce LR at stage transition
            if self.global_step == int(self.config.prog_stage2_pct * self.total_steps):
                logger.info("Entering full ternary stage, reducing learning rate")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.7
        
        return epoch_loss / epoch_steps
    
    def validate(self):
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", disable=self.rank != 0):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    training=False
                )
                logits = outputs['logits']
                
                loss = self.criterion(
                    logits.view(-1, self.config.vocab_size),
                    batch['labels'].view(-1)
                )
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        perplexity = np.exp(avg_loss)
        
        # Prepare validation metrics
        val_metrics = {
            'val/loss': avg_loss,
            'val/perplexity': perplexity,
        }
        
        # Log validation metrics
        self.log_metrics(val_metrics, self.global_step)
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return {'loss': avg_loss, 'perplexity': perplexity}
    
    def clip_quantizer_gradients(self):
        """Apply LSQ gradient clipping to quantizer parameters."""
        model = self.model.module if self.distributed else self.model
        
        for module in model.modules():
            if hasattr(module, 's_tilde'):  # TTQ quantizer
                if module.s_tilde.grad is not None:
                    module.s_tilde.grad.clamp_(-self.config.lsq_grad_clip, self.config.lsq_grad_clip)
                if hasattr(module, 't') and module.t.grad is not None:
                    module.t.grad.clamp_(-self.config.lsq_grad_clip, self.config.lsq_grad_clip)
                if hasattr(module, 'delta') and module.delta.grad is not None:
                    module.delta.grad.clamp_(-self.config.lsq_grad_clip, self.config.lsq_grad_clip)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'args': self.args
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch}, step {self.global_step})")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.epoch, self.args.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            epoch_loss = self.train_epoch()
            
            # Validation at end of epoch
            val_metrics = self.validate()
            
            # Log epoch metrics
            if self.rank == 0:
                logger.info(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val Perplexity: {val_metrics['perplexity']:.2f}")
                
                # Save epoch checkpoint
                self.save_checkpoint(f'epoch_{epoch}')
        
        # Post-training calibration
        if self.rank == 0:
            logger.info("Running post-training QDyT calibration...")
            model = self.model.module if self.distributed else self.model
            calibrate_qdyt_norms(model, self.val_loader, num_batches=100)
            
            # Save final model
            self.save_checkpoint('final')
            
            # Final evaluation
            logger.info("Running final evaluation...")
            test_metrics = self.validate()
            logger.info(f"Final Test - Loss: {test_metrics['loss']:.4f}, "
                       f"Perplexity: {test_metrics['perplexity']:.2f}")
        
        if self.writer:
            self.writer.close()
        
        # Close wandb run
        if self.wandb_run and self.rank == 0:
            self.wandb_run.finish()
            logger.info("Wandb run finished")
        
        logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train BitNet-QDyT-v2 model')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='base',
                      choices=['small', 'base', 'large', 'rtx'],
                      help='Model size configuration')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='wikitext-103-v1',
                      choices=['wikitext-2-v1', 'wikitext-103-v1'],
                      help='WikiText dataset to use')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                      help='Tokenizer to use')
    parser.add_argument('--max_length', type=int, default=512,
                      help='Maximum sequence length')
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                      help='Masked language modeling probability')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                      help='Cache directory for datasets')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=10000,
                      help='Warmup steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                      help='Gradient clipping')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                      help='Gradient accumulation steps')
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['linear', 'cosine'],
                      help='Learning rate scheduler')
    
    # Optimization arguments
    parser.add_argument('--fp16', action='store_true',
                      help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    
    # Logging arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='Output directory')
    parser.add_argument('--log_interval', type=int, default=100,
                      help='Logging interval')
    parser.add_argument('--eval_interval', type=int, default=1000,
                      help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=5000,
                      help='Checkpoint save interval')
    
    # Resume training
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Resume from checkpoint')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='bitnet-qdyt-v2',
                      help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                      help='Wandb run name (auto-generated if not provided)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                      help='Wandb tags for the run')
    parser.add_argument('--wandb_notes', type=str, default=None,
                      help='Notes for the wandb run')
    parser.add_argument('--disable_wandb', action='store_true',
                      help='Disable wandb logging')
    parser.add_argument('--wandb_mode', type=str, default='online',
                      choices=['online', 'offline', 'disabled'],
                      help='Wandb mode (online, offline, or disabled)')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()