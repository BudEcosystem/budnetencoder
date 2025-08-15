#!/bin/bash

# RTX 3080 optimized training launch script
# Specifically tuned for 10GB VRAM with best performance settings

set -e  # Exit on any error

echo "=================================================="
echo "BitNet-QDyT-v2 RTX 3080 Training Launch"
echo "=================================================="

# Check GPU
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Set environment variables for RTX optimization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set memory growth to prevent OOM
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="rtx_training_${TIMESTAMP}"

echo "Starting training with optimized RTX settings..."
echo "Run name: $RUN_NAME"
echo "Output directory: ./outputs"
echo "Model configuration: RTX 3080 optimized (~139M params)"
echo ""

# Training arguments optimized for RTX 3080
python3 train.py \
    --model_size rtx \
    --dataset wikitext-103-v1 \
    --batch_size 24 \
    --learning_rate 2e-4 \
    --num_epochs 15 \
    --warmup_steps 5000 \
    --max_length 384 \
    --output_dir ./outputs \
    --fp16 \
    --num_workers 2 \
    --log_interval 50 \
    --eval_interval 500 \
    --save_interval 2000 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --mlm_probability 0.15 \
    --cache_dir ./cache \
    2>&1 | tee "./outputs/logs/rtx_training_${TIMESTAMP}.log"

echo ""
echo "=================================================="
echo "Training completed!"
echo "Log file: ./outputs/logs/rtx_training_${TIMESTAMP}.log"
echo "Checkpoints: ./outputs/checkpoints/"
echo "=================================================="