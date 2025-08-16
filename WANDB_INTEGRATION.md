# Weights & Biases Integration for BitNet-QDyT-v2

This document describes the comprehensive Weights & Biases (wandb) integration added to the BitNet-QDyT-v2 training pipeline.

## Features

### 🔧 Core Integration
- **Automatic experiment tracking** with configurable project names and run names
- **Offline mode support** for environments without internet access
- **Robust error handling** with graceful fallbacks
- **Multi-process support** (only main process logs to wandb)

### 📊 Comprehensive Metrics Logging
- **Training metrics**: loss, learning rate, training stage, epoch, global step
- **Validation metrics**: validation loss, perplexity
- **System metrics**: CPU usage, memory usage, GPU utilization, GPU memory, GPU temperature
- **Model metrics**: gradient norms, quantization parameters
- **Progressive quantization**: QDrop probability, training stage transitions

### 🏗️ Model & Experiment Tracking
- **Hyperparameter logging**: All training arguments automatically tracked
- **Model architecture**: Parameter counts, layer dimensions, configuration
- **Dataset information**: Dataset size, batch counts, tokenizer details
- **Hardware information**: Device type, world size for distributed training

## Usage

### Basic Usage

```bash
# Train with wandb integration (offline mode)
python3 train.py --model_size rtx --dataset wikitext-103-v1 --wandb_mode offline

# Train with custom wandb project and run name
python3 train.py \
    --model_size base \
    --dataset wikitext-2-v1 \
    --wandb_project "my-bitnet-experiments" \
    --wandb_run_name "baseline-experiment" \
    --wandb_tags "baseline" "test"
```

### Enhanced Launch Script

Use the provided launch script for production training with wandb:

```bash
./launch_with_wandb.sh
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb_project` | `"bitnet-qdyt-v2"` | Wandb project name |
| `--wandb_run_name` | Auto-generated | Custom run name |
| `--wandb_tags` | `[]` | Tags for the run |
| `--wandb_notes` | `None` | Run description/notes |
| `--wandb_mode` | `"online"` | Mode: online, offline, disabled |
| `--disable_wandb` | `False` | Completely disable wandb |

## Wandb Modes

### Online Mode (Default)
- Requires wandb account and internet connection
- Real-time syncing to wandb cloud
- Interactive dashboard access

```bash
python3 train.py --wandb_mode online
```

### Offline Mode
- No internet connection required
- Data stored locally in `./wandb/` directory
- Can sync later with `wandb sync`

```bash
python3 train.py --wandb_mode offline
```

### Disabled Mode
- Completely disables wandb logging
- Falls back to TensorBoard only

```bash
python3 train.py --wandb_mode disabled
# OR
python3 train.py --disable_wandb
```

## Metrics Dashboard

The wandb integration tracks the following metrics:

### Training Metrics
```
train/loss                    # Training loss per step
train/learning_rate          # Current learning rate
train/qdrop_probability      # Current QDrop probability
train/training_stage         # warmup/partial_ternary/full_ternary
train/epoch                  # Current epoch
train/global_step           # Global training step
train/grad_norm             # Gradient norm
```

### Validation Metrics
```
val/loss                     # Validation loss
val/perplexity              # Validation perplexity
```

### System Metrics
```
cpu_percent                  # CPU utilization %
memory_percent              # RAM usage %
gpu_utilization             # GPU utilization %
gpu_memory_used             # GPU memory used (MB)
gpu_memory_percent          # GPU memory usage %
gpu_temperature             # GPU temperature (°C)
torch_gpu_memory_allocated  # PyTorch GPU memory (GB)
torch_gpu_memory_reserved   # PyTorch reserved GPU memory (GB)
```

## Configuration Tracking

All training hyperparameters are automatically logged:

```python
{
    "model_size": "rtx",
    "dataset": "wikitext-103-v1", 
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_epochs": 15,
    "warmup_steps": 5000,
    "max_length": 384,
    "fp16": True,
    "gradient_accumulation_steps": 3,
    "total_parameters": 94567890,
    "trainable_parameters": 94567890,
    "vocab_size": 30522,
    "hidden_size": 768,
    "steps_per_epoch": 106206,
    "total_training_steps": 1593090
}
```

## Syncing Offline Runs

If training in offline mode, sync data to wandb cloud:

```bash
# Sync specific run
wandb sync ./wandb/offline-run-20231201_120000-abc123

# Sync all offline runs
wandb sync ./wandb/offline-run-*
```

## Examples

### Production Training with Full Logging
```bash
python3 train.py \
    --model_size rtx \
    --dataset wikitext-103-v1 \
    --batch_size 8 \
    --gradient_accumulation_steps 3 \
    --wandb_project "bitnet-production" \
    --wandb_run_name "rtx-3080-optimized" \
    --wandb_tags "production" "rtx-3080" "optimized" \
    --wandb_notes "Production training on RTX 3080 with memory optimizations" \
    --wandb_mode offline
```

### Experimental Run with Custom Tags
```bash
python3 train.py \
    --model_size base \
    --dataset wikitext-2-v1 \
    --learning_rate 1e-4 \
    --wandb_project "bitnet-experiments" \
    --wandb_run_name "lr-ablation-1e4" \
    --wandb_tags "ablation" "learning-rate" "baseline" \
    --wandb_notes "Learning rate ablation study - testing 1e-4"
```

### Quick Test without Wandb
```bash
python3 train.py \
    --model_size small \
    --dataset wikitext-2-v1 \
    --num_epochs 1 \
    --disable_wandb
```

## Troubleshooting

### Common Issues

1. **Authentication Error**: Use offline mode if no wandb account
   ```bash
   python3 train.py --wandb_mode offline
   ```

2. **Network Issues**: Automatically falls back to offline mode
   
3. **Disk Space**: Offline runs store data locally in `./wandb/`

### Debugging

Enable verbose logging:
```bash
export WANDB_SILENT=false
python3 train.py --wandb_mode offline
```

## Integration Details

The wandb integration is implemented in the `Trainer` class:

- `setup_wandb()`: Initialize wandb with error handling
- `get_system_metrics()`: Collect system and GPU metrics
- `log_metrics()`: Log to both wandb and TensorBoard
- Automatic configuration and model architecture logging
- Graceful shutdown with `wandb.finish()`

The integration is designed to be:
- **Non-intrusive**: Training continues normally if wandb fails
- **Flexible**: Multiple modes and configuration options
- **Comprehensive**: Logs all relevant metrics for analysis
- **Production-ready**: Robust error handling and fallbacks

## Files Added/Modified

- `train.py`: Added wandb integration with comprehensive logging
- `launch_with_wandb.sh`: Production launch script with wandb
- `test_wandb.py`: Test script for wandb integration
- `WANDB_INTEGRATION.md`: This documentation

The integration maintains backward compatibility and doesn't break existing workflows.