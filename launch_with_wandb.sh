#!/bin/bash

# BitNet-QDyT-v2 Training with Weights & Biases Integration
# Enhanced training script with comprehensive experiment tracking

set -e

echo "========================================================="
echo "BitNet-QDyT-v2 Training with Wandb Integration"
echo "========================================================="

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader
echo ""

# Memory optimization environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false

# Clear GPU cache
echo "Clearing GPU cache..."
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting training with Wandb integration..."
echo "Settings:"
echo "  - Model size: RTX optimized"
echo "  - Dataset: WikiText-103"
echo "  - Batch size: 8 (with gradient accumulation 3)"
echo "  - Effective batch size: 24"
echo "  - Wandb project: bitnet-qdyt-v2"
echo "  - Run name: rtx-training-${TIMESTAMP}"
echo ""

# Enhanced training with wandb integration
python3 train.py \
    --model_size rtx \
    --dataset wikitext-103-v1 \
    --batch_size 8 \
    --gradient_accumulation_steps 3 \
    --learning_rate 2e-4 \
    --num_epochs 15 \
    --warmup_steps 5000 \
    --max_length 384 \
    --output_dir ./outputs \
    --fp16 \
    --num_workers 1 \
    --log_interval 50 \
    --eval_interval 1000 \
    --save_interval 3000 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --mlm_probability 0.15 \
    --cache_dir ./cache \
    --wandb_project "bitnet-qdyt-v2" \
    --wandb_run_name "rtx-training-${TIMESTAMP}" \
    --wandb_tags "rtx" "production" "wikitext-103" \
    --wandb_notes "Production training run on RTX 3080 with optimized memory settings" \
    --wandb_mode "offline" \
    2>&1 | tee "./outputs/logs/wandb_training_${TIMESTAMP}.log"

echo ""
echo "========================================================="
echo "Training completed!"
echo "Log file: ./outputs/logs/wandb_training_${TIMESTAMP}.log"
echo "Checkpoints: ./outputs/checkpoints/"
echo "Wandb data: ./wandb/"
echo ""
echo "To sync wandb data to cloud (if you have wandb account):"
echo "  wandb sync ./wandb/offline-run-*"
echo "========================================================="