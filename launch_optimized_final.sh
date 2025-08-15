#!/bin/bash

# FINAL OPTIMIZED training script for RTX 3080
# Conservative settings to ensure stable training

set -e

echo "=================================================="
echo "BitNet-QDyT-v2 Training - FINAL OPTIMIZED"
echo "=================================================="

# Clear any existing Python processes
echo "Clearing any stuck processes..."
pkill -f "python3 train.py" 2>/dev/null || true
sleep 2

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

echo "Starting OPTIMIZED training..."
echo "Settings:"
echo "  - Batch size: 8 (reduced from 24)"
echo "  - Gradient accumulation: 3 steps"
echo "  - Effective batch size: 24"
echo "  - Workers: 1 (reduced for stability)"
echo ""

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
    2>&1 | tee "./outputs/logs/optimized_${TIMESTAMP}.log"