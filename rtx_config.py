"""
Optimized configuration for RTX 3080 (10GB VRAM) training.
Balanced settings for best performance on this hardware.
"""

from model_configs import BitNetQDyTBaseConfig

class RTXOptimizedConfig(BitNetQDyTBaseConfig):
    """RTX 3080 optimized configuration for ~100M parameters."""
    
    def __init__(self):
        super().__init__()
        
        # RTX 3080 specific optimizations
        self.vocab_size = 30522
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 2304  # Reduced from 3072 to 2304 (3x) for memory efficiency
        self.max_position_embeddings = 384  # Reduced from 512 for memory efficiency
        
        # Quantization optimized for RTX 3080
        self.weight_bits = 2  # Ternary
        self.activation_bits = 4
        self.qk_bits = 6  # Slightly lower precision for memory efficiency
        self.gate_bits = 8
        self.embedding_bits = 6  # Reduced for memory efficiency
        self.lm_head_bits = 6   # Reduced for memory efficiency
        self.mixed_precision_last_blocks = 1  # Only last block gets higher precision
        self.vo_bits = 6
        
        # Memory-efficient orthogonal mixing
        self.block_hadamard_size = 64
        self.use_dpd = True
        self.use_block_hadamard = True
        self.alternate_ffn_mixing = True
        
        # QDyT settings optimized for smaller batches
        self.qdyt_group_size = 8  # Smaller groups for efficiency
        self.qdyt_alpha_init = 0.1
        self.qdyt_alpha_target = 0.4
        self.qdyt_warmup_steps = 3000  # Shorter warmup
        self.use_pact_clip = True
        
        # Percentile scaling
        self.percentile_value = 99.5  # Slightly more conservative
        self.percentile_ema_momentum = 0.9  # Faster adaptation
        self.use_stochastic_rounding = True
        
        # TTQ/LSQ+ settings
        self.ttq_init_percentile = 25.0  # More conservative initialization
        self.ttq_delta_factor = 0.03
        self.lsq_grad_clip = 0.8
        self.use_dither = True
        self.ternary_dropout_prob = 0.05
        self.ternary_dropout_steps = 15000
        
        # Aggressive progressive quantization for faster training
        self.progressive_quantization = True
        self.prog_stage1_pct = 0.05  # Very short warmup (5%)
        self.prog_stage2_pct = 0.2   # Quick transition to full ternary (20%)
        self.qdrop_prob_initial = 0.15
        self.qdrop_anneal_steps = 25000
        
        # Residual path
        self.use_skip_init = True
        self.skip_init_scale = 0.4  # Slightly more aggressive
        
        # No teacher distillation to save memory
        self.use_teacher_guidance = False


def get_rtx_training_args():
    """Get optimized training arguments for RTX 3080."""
    return {
        'model_size': 'rtx',  # Use our custom config
        'batch_size': 24,     # Optimized for 10GB VRAM
        'learning_rate': 2e-4, # Slightly lower for stability
        'num_epochs': 15,     # More epochs with smaller model
        'warmup_steps': 5000, # Shorter warmup
        'max_length': 384,    # Shorter sequences for memory efficiency
        'gradient_accumulation_steps': 2,  # Effective batch size = 48
        'fp16': True,         # Essential for RTX cards
        'dataloader_num_workers': 2,  # Conservative for stability
        'save_interval': 2000,
        'eval_interval': 500,
        'log_interval': 50,
    }


def estimate_memory_usage():
    """Estimate memory usage for RTX configuration."""
    config = RTXOptimizedConfig()
    
    # Rough memory estimation
    params = (
        config.vocab_size * config.hidden_size +  # embeddings
        config.max_position_embeddings * config.hidden_size +
        config.type_vocab_size * config.hidden_size +
        config.num_hidden_layers * (
            4 * config.hidden_size * config.hidden_size +  # attention
            2 * config.hidden_size * config.intermediate_size +  # ffn
            config.intermediate_size * config.hidden_size
        ) +
        config.vocab_size * config.hidden_size  # lm_head
    )
    
    # Memory breakdown
    model_memory = params * 4 / 1e9  # FP32 bytes to GB
    activation_memory = 24 * 384 * config.hidden_size * 4 / 1e9  # batch * seq * hidden * fp32
    optimizer_memory = params * 8 / 1e9  # AdamW states
    
    total_memory = model_memory + activation_memory + optimizer_memory
    
    print(f"Estimated memory usage for RTX configuration:")
    print(f"  Model: {model_memory:.2f} GB")
    print(f"  Activations: {activation_memory:.2f} GB")
    print(f"  Optimizer: {optimizer_memory:.2f} GB")
    print(f"  Total: {total_memory:.2f} GB")
    print(f"  RTX 3080 VRAM: 10.0 GB")
    print(f"  Safety margin: {10.0 - total_memory:.2f} GB")
    
    return total_memory < 8.5  # Leave 1.5GB safety margin


if __name__ == "__main__":
    config = RTXOptimizedConfig()
    args = get_rtx_training_args()
    
    print("RTX 3080 Optimized Configuration:")
    print(f"  Parameters: ~{((config.vocab_size + config.max_position_embeddings + config.type_vocab_size) * config.hidden_size + config.num_hidden_layers * (4 * config.hidden_size**2 + 3 * config.hidden_size * config.intermediate_size) + config.vocab_size * config.hidden_size) / 1e6:.1f}M")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  FFN size: {config.intermediate_size}")
    print(f"  Max length: {config.max_position_embeddings}")
    print(f"  Batch size: {args['batch_size']}")
    
    # Check memory feasibility
    is_feasible = estimate_memory_usage()
    print(f"\nMemory feasible: {'✓' if is_feasible else '✗'}")