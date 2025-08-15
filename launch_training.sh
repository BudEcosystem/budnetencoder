#!/bin/bash

# Launch script for BitNet-QDyT-v2 training
# Supports both single-GPU and multi-GPU training

# Default values
MODEL_SIZE="base"  # 100M parameters
DATASET="wikitext-103-v1"
BATCH_SIZE=32
LEARNING_RATE=3e-4
NUM_EPOCHS=10
OUTPUT_DIR="./outputs"
FP16=false
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fp16)
            FP16=true
            shift
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model_size SIZE      Model size (small/base/large, default: base)"
            echo "  --dataset DATASET      Dataset name (default: wikitext-103-v1)"
            echo "  --batch_size SIZE      Batch size per GPU (default: 32)"
            echo "  --learning_rate LR     Learning rate (default: 3e-4)"
            echo "  --num_epochs N         Number of epochs (default: 10)"
            echo "  --output_dir DIR       Output directory (default: ./outputs)"
            echo "  --fp16                 Use mixed precision training"
            echo "  --resume PATH          Resume from checkpoint"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Set environment variables for better performance
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "Detected $NUM_GPUS GPU(s)"

# Build training command
TRAINING_ARGS="
    --model_size $MODEL_SIZE \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --warmup_steps 10000 \
    --log_interval 100 \
    --eval_interval 1000 \
    --save_interval 5000 \
    --num_workers 4 \
    --grad_clip 1.0 \
    --weight_decay 0.01 \
    --scheduler cosine \
    --max_length 512 \
    --mlm_probability 0.15 \
    --cache_dir ./cache
"

# Add optional arguments
if [ "$FP16" = true ]; then
    TRAINING_ARGS="$TRAINING_ARGS --fp16"
fi

if [ -n "$RESUME" ]; then
    TRAINING_ARGS="$TRAINING_ARGS --resume_from $RESUME"
fi

# Launch training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training on $NUM_GPUS GPUs..."
    echo "Training command: torchrun --nproc_per_node=$NUM_GPUS train.py $TRAINING_ARGS"
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train.py $TRAINING_ARGS 2>&1 | tee "$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
else
    echo "Launching single-GPU training..."
    echo "Training command: python train.py $TRAINING_ARGS"
    
    python train.py $TRAINING_ARGS 2>&1 | tee "$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Training completed. Results saved to $OUTPUT_DIR"