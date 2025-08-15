"""
Evaluation utilities for BitNet-QDyT-v2 model.
Includes perplexity calculation, accuracy metrics, and model analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from implementation import BitNetQDyTModel


def calculate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device('cuda')
) -> float:
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
    
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                training=False
            )
            logits = outputs['logits']
            
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1)
            )
            
            # Only count non-ignored tokens
            mask = batch['labels'].view(-1) != -100
            total_loss += loss[mask].sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device('cuda'),
    top_k: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
        top_k: List of k values for top-k accuracy
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = defaultdict(float)
    total_predictions = 0
    correct_predictions = {k: 0 for k in top_k}
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                training=False
            )
            logits = outputs['logits']
            
            # Loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1)
            )
            metrics['loss'] += loss.item()
            
            # Top-k accuracy
            mask = batch['labels'] != -100
            if mask.any():
                masked_logits = logits[mask]
                masked_labels = batch['labels'][mask]
                
                for k in top_k:
                    _, top_k_preds = masked_logits.topk(k, dim=-1)
                    correct = (top_k_preds == masked_labels.unsqueeze(-1)).any(dim=-1)
                    correct_predictions[k] += correct.sum().item()
                
                total_predictions += mask.sum().item()
    
    # Calculate final metrics
    num_batches = len(dataloader)
    metrics['loss'] /= num_batches
    metrics['perplexity'] = np.exp(metrics['loss'])
    
    for k in top_k:
        metrics[f'top_{k}_accuracy'] = correct_predictions[k] / total_predictions if total_predictions > 0 else 0
    
    return dict(metrics)


def analyze_quantization_stats(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device('cuda'),
    num_batches: int = 10
) -> Dict[str, Dict]:
    """
    Analyze quantization statistics of the model.
    
    Args:
        model: Model to analyze
        dataloader: DataLoader for analysis
        device: Device to run on
        num_batches: Number of batches to analyze
    
    Returns:
        Dictionary of quantization statistics
    """
    model.eval()
    stats = defaultdict(lambda: defaultdict(list))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass to populate statistics
            _ = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                training=False
            )
            
            # Collect statistics from quantizers
            for name, module in model.named_modules():
                # TTQ quantizer stats
                if hasattr(module, 's_tilde'):
                    s = torch.nn.functional.softplus(module.s_tilde)
                    stats['ternary_scales'][name].append(s.mean().item())
                    stats['ternary_thresholds'][name].append(module.t.mean().item())
                
                # Activation quantizer stats
                if hasattr(module, 'percentile_vals'):
                    stats['activation_scales'][name].append(module.percentile_vals.mean().item())
                    stats['activation_utilization'][name].append(
                        (module.percentile_vals > 0).float().mean().item()
                    )
                
                # QDyT norm stats
                if hasattr(module, 'alpha'):
                    stats['qdyt_alpha'][name].append(module.alpha.item())
                    if hasattr(module, 'a_tilde'):
                        a = torch.nn.functional.softplus(module.a_tilde)
                        stats['qdyt_clip'][name].append(a.item())
    
    # Average statistics
    averaged_stats = {}
    for stat_type, layer_stats in stats.items():
        averaged_stats[stat_type] = {}
        for layer_name, values in layer_stats.items():
            averaged_stats[stat_type][layer_name] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    return averaged_stats


def analyze_attention_patterns(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device('cuda'),
    num_samples: int = 5
) -> Dict[str, np.ndarray]:
    """
    Analyze attention patterns in the model.
    
    Args:
        model: Model to analyze
        dataloader: DataLoader for analysis
        device: Device to run on
        num_samples: Number of samples to analyze
    
    Returns:
        Dictionary of attention patterns
    """
    model.eval()
    attention_patterns = defaultdict(list)
    
    # Hook to capture attention weights
    attention_weights = {}
    
    def attention_hook(module, input, output, layer_name):
        # Assuming attention weights are computed in the module
        # This is a simplified version - actual implementation depends on model architecture
        attention_weights[layer_name] = output
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if 'attention' in name.lower() and hasattr(module, 'forward'):
            hook = module.register_forward_hook(
                lambda m, i, o, n=name: attention_hook(m, i, o, n)
            )
            hooks.append(hook)
    
    with torch.no_grad():
        for sample_idx, batch in enumerate(dataloader):
            if sample_idx >= num_samples:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            _ = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                training=False
            )
            
            # Store attention patterns
            for layer_name, weights in attention_weights.items():
                attention_patterns[layer_name].append(weights.cpu().numpy())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average patterns
    averaged_patterns = {}
    for layer_name, patterns in attention_patterns.items():
        if patterns:
            averaged_patterns[layer_name] = np.mean(patterns, axis=0)
    
    return averaged_patterns


def plot_training_curves(
    log_file: str,
    output_dir: str = './plots'
) -> None:
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to training log file
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load training logs
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(logs['train_loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    
    # Validation loss
    axes[0, 1].plot(logs['val_loss'])
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    
    # Learning rate
    axes[1, 0].plot(logs['learning_rate'])
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('LR')
    
    # Perplexity
    axes[1, 1].plot(logs['perplexity'])
    axes[1, 1].set_title('Perplexity')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Perplexity')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png')
    plt.close()


def save_evaluation_results(
    results: Dict,
    output_file: str
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Dictionary of evaluation results
        output_file: Path to output file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved evaluation results to {output_file}")


def compare_models(
    model_paths: List[str],
    dataloader: DataLoader,
    device: torch.device = torch.device('cuda')
) -> Dict[str, Dict]:
    """
    Compare multiple model checkpoints.
    
    Args:
        model_paths: List of paths to model checkpoints
        dataloader: DataLoader for evaluation
        device: Device to run on
    
    Returns:
        Dictionary of comparison results
    """
    results = {}
    
    for path in model_paths:
        print(f"Evaluating {path}...")
        
        # Load model
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = BitNetQDyTModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate
        metrics = evaluate_model(model, dataloader, device)
        results[Path(path).stem] = metrics
    
    return results


def print_evaluation_summary(results: Dict[str, float]) -> None:
    """
    Print formatted evaluation summary.
    
    Args:
        results: Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:20s}: {value:.4f}")
        else:
            print(f"{metric:20s}: {value}")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    from data_processing import load_wikitext_dataset
    from model_configs import get_model_config
    
    parser = argparse.ArgumentParser(description='Evaluate BitNet-QDyT-v2 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='wikitext-2-v1',
                      help='Dataset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                      help='Output directory for results')
    parser.add_argument('--analyze_quantization', action='store_true',
                      help='Analyze quantization statistics')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint['config']
    model = BitNetQDyTModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Load data
    _, val_loader, test_loader, tokenizer = load_wikitext_dataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size
    )
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device)
    print_evaluation_summary(results)
    
    # Save results
    save_evaluation_results(results, f"{args.output_dir}/evaluation_results.json")
    
    # Analyze quantization if requested
    if args.analyze_quantization:
        print("Analyzing quantization statistics...")
        quant_stats = analyze_quantization_stats(model, val_loader, device)
        save_evaluation_results(quant_stats, f"{args.output_dir}/quantization_stats.json")