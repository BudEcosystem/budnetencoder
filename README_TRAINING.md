# BitNet-QDyT-v2 Training Guide

This guide provides instructions for training the BitNet-QDyT-v2 model with ~100M parameters on the WikiText dataset.

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install core dependencies manually:
```bash
pip install torch numpy transformers datasets tqdm tensorboard matplotlib
```

### 2. Run Setup Script

```bash
python3 setup.py
```

This will:
- Install all dependencies
- Create necessary directories
- Check CUDA availability
- Download tokenizer

## 🚀 Quick Start

### Single GPU Training

```bash
python3 train.py \
    --model_size base \
    --dataset wikitext-103-v1 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --output_dir ./outputs
```

### Multi-GPU Training

```bash
./launch_training.sh \
    --model_size base \
    --dataset wikitext-103-v1 \
    --batch_size 32 \
    --num_epochs 10
```

### Mixed Precision Training (Faster)

Add `--fp16` flag for mixed precision training:

```bash
./launch_training.sh \
    --model_size base \
    --dataset wikitext-103-v1 \
    --fp16
```

## 📊 Model Configurations

| Model Size | Parameters | Hidden Size | Layers | Heads | FFN Size |
|------------|------------|-------------|--------|-------|----------|
| small      | ~25M       | 512         | 8      | 8     | 2048     |
| **base**   | **~100M**  | **768**     | **12** | **12**| **3072** |
| large      | ~350M      | 1024        | 24     | 16    | 4096     |

The **base** configuration is optimized for ~100M parameters as requested.

## 🎯 Training Parameters

### Recommended Settings for 100M Model

```bash
--model_size base          # 100M parameter model
--batch_size 32            # Adjust based on GPU memory
--learning_rate 3e-4       # Starting learning rate
--num_epochs 10            # Full training
--warmup_steps 10000       # Warmup period
--fp16                     # Mixed precision (recommended)
```

### GPU Memory Requirements

- **Small model (25M)**: ~4GB VRAM
- **Base model (100M)**: ~8-12GB VRAM
- **Large model (350M)**: ~16-24GB VRAM

Adjust batch size if you encounter OOM errors:
- 16GB GPU: batch_size=32
- 12GB GPU: batch_size=16-24
- 8GB GPU: batch_size=8-16

## 📈 Training Stages

The model uses progressive quantization with three stages:

1. **Warmup (0-10%)**: Int8 weights, QDyT normalization warmup
2. **Partial Ternary (10-30%)**: Top half layers ternarized
3. **Full Ternary (30-100%)**: All layers ternarized, reduced LR

## 🔄 Resume Training

To resume from a checkpoint:

```bash
./launch_training.sh \
    --resume outputs/checkpoints/best.pt \
    --model_size base
```

## 📝 Monitoring

Training progress is logged to:
- TensorBoard: `outputs/logs/`
- Console output: `outputs/logs/training_*.log`
- Checkpoints: `outputs/checkpoints/`

View TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

## 🧪 Evaluation

Evaluate a trained model:

```bash
python3 evaluate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --dataset wikitext-2-v1 \
    --batch_size 32
```

## 🚨 Troubleshooting

### CUDA Not Available
- Check GPU drivers: `nvidia-smi`
- Verify PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory (OOM)
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision (`--fp16`)

### Slow Training
- Enable mixed precision
- Increase batch size (if memory allows)
- Use multiple GPUs
- Check data loading workers (`--num_workers`)

## 📂 Project Structure

```
budnetencoder/
├── implementation.py      # Core model implementation
├── model_configs.py       # Model size configurations
├── data_processing.py     # WikiText data loading
├── train.py              # Training script
├── evaluate.py           # Evaluation utilities
├── launch_training.sh    # Multi-GPU launcher
├── setup.py             # Environment setup
├── requirements.txt     # Dependencies
└── run_quick_test.py    # Quick test script
```

## 🎓 Training Tips

1. **Start with small dataset**: Test with `wikitext-2-v1` first
2. **Monitor loss curves**: Check for convergence in TensorBoard
3. **Save checkpoints regularly**: Use `--save_interval` flag
4. **Adjust learning rate**: Reduce if loss plateaus
5. **Use validation set**: Monitor overfitting

## 💾 Expected Training Time

On a single V100 GPU:
- Small model: ~2-4 hours
- **Base model (100M)**: ~8-12 hours
- Large model: ~24-36 hours

Training time scales linearly with dataset size and inversely with number of GPUs.

## 🔍 Model Analysis

After training, analyze quantization statistics:

```bash
python3 evaluate.py \
    --checkpoint outputs/checkpoints/final.pt \
    --analyze_quantization \
    --output_dir eval_results
```

## 📊 Expected Results

With proper training on WikiText-103:
- Perplexity: ~25-35 (depending on quantization)
- Training loss: ~2.5-3.5
- Validation loss: ~3.0-4.0

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review training logs in `outputs/logs/`
3. Verify CUDA and dependencies are properly installed

## 📜 License

This implementation follows the BitNet-QDyT-v2 paper specifications and is provided for research purposes.