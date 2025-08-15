#!/bin/bash

# RTX 3080 MEMORY-OPTIMIZED training launch script
# Reduced batch size and optimized settings for 10GB VRAM

set -e

echo "=================================================="
echo "BitNet-QDyT-v2 RTX 3080 Training (Memory Optimized)"
echo "=================================================="

# Check GPU
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Set environment variables for memory optimization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

# Enable memory efficient attention
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.6"

# Clear cache
echo "Clearing GPU cache..."
python3 -c "import torch; torch.cuda.empty_cache()"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="rtx_optimized_${TIMESTAMP}"

echo "Starting training with MEMORY OPTIMIZED settings..."
echo "Run name: $RUN_NAME"
echo "Output directory: ./outputs"
echo "Batch size: 8 (reduced from 24)"
echo "Gradient accumulation: 3 (effective batch = 24)"
echo ""

# REDUCED BATCH SIZE AND OPTIMIZED SETTINGS
python3 train.py \
    --model_size rtx \
    --dataset wikitext-103-v1 \
    --batch_size 8 \
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
    2>&1 | tee "./outputs/logs/rtx_optimized_${TIMESTAMP}.log"

echo ""
echo "=================================================="
echo "Training completed!"
echo "Log file: ./outputs/logs/rtx_optimized_${TIMESTAMP}.log"
echo "Checkpoints: ./outputs/checkpoints/"
echo "==================================================