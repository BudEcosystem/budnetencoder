"""
Model configurations for different BitNet-QDyT-v2 model sizes.
Includes configurations optimized for ~100M parameters.
"""

from dataclasses import dataclass
from implementation import BitNetQDyTConfig


@dataclass
class BitNetQDyTSmallConfig(BitNetQDyTConfig):
    """Small model config (~25M parameters)"""
    def __init__(self):
        super().__init__(
            vocab_size=30522,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,  # 4x expansion
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )


@dataclass
class BitNetQDyTBaseConfig(BitNetQDyTConfig):
    """Base model config (~100M parameters)"""
    def __init__(self):
        super().__init__(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,  # 4x expansion for 100M params
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            
            # Quantization settings optimized for 100M model
            weight_bits=2,  # Ternary
            activation_bits=4,
            qk_bits=6,  # Slightly lower precision for 100M model
            gate_bits=8,
            embedding_bits=8,
            lm_head_bits=8,
            mixed_precision_last_blocks=2,
            vo_bits=6,
            
            # Orthogonal mixing
            block_hadamard_size=64,
            use_dpd=True,
            use_block_hadamard=True,
            alternate_ffn_mixing=True,
            
            # QDyT normalization
            qdyt_group_size=16,
            qdyt_alpha_init=0.05,
            qdyt_alpha_target=0.5,
            qdyt_warmup_steps=5000,
            use_pact_clip=True,
            
            # Percentile scaling
            percentile_value=99.7,
            percentile_ema_momentum=0.95,
            use_stochastic_rounding=True,
            
            # TTQ/LSQ+ settings
            ttq_init_percentile=30.0,
            ttq_delta_factor=0.05,
            lsq_grad_clip=1.0,
            use_dither=True,
            ternary_dropout_prob=0.05,
            ternary_dropout_steps=20000,
            
            # Training settings
            progressive_quantization=True,
            prog_stage1_pct=0.1,
            prog_stage2_pct=0.3,
            qdrop_prob_initial=0.1,
            qdrop_anneal_steps=50000,
            
            # Residual path
            use_skip_init=True,
            skip_init_scale=0.5,
            
            # Teacher distillation
            use_teacher_guidance=False,
            kd_temperature=3.0,
            kd_weight_max=0.3
        )


@dataclass
class BitNetQDyTLargeConfig(BitNetQDyTConfig):
    """Large model config (~350M parameters)"""
    def __init__(self):
        super().__init__(
            vocab_size=30522,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,  # 4x expansion
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            
            # Adjusted quantization for larger model
            weight_bits=2,
            activation_bits=4,
            qk_bits=8,
            gate_bits=8,
            embedding_bits=8,
            lm_head_bits=8,
            mixed_precision_last_blocks=3,
            vo_bits=8,
            
            # Larger blocks for efficiency
            block_hadamard_size=128,
            qdyt_group_size=32,
            
            # More aggressive quantization schedule
            prog_stage1_pct=0.05,
            prog_stage2_pct=0.25,
            qdrop_anneal_steps=30000
        )


def get_model_config(model_size: str = "base") -> BitNetQDyTConfig:
    """
    Get model configuration by size name.
    
    Args:
        model_size: One of "small", "base", "large", "rtx"
    
    Returns:
        BitNetQDyTConfig instance
    """
    configs = {
        "small": BitNetQDyTSmallConfig(),
        "base": BitNetQDyTBaseConfig(),
        "large": BitNetQDyTLargeConfig()
    }
    
    # Import RTX config if requested
    if model_size == "rtx":
        from rtx_config import RTXOptimizedConfig
        return RTXOptimizedConfig()
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys()) + ['rtx']}")
    
    return configs[model_size]


def calculate_model_params(config: BitNetQDyTConfig) -> dict:
    """
    Calculate approximate parameter count for a given configuration.
    
    Args:
        config: Model configuration
    
    Returns:
        Dictionary with parameter counts by component
    """
    params = {}
    
    # Embeddings
    params['word_embeddings'] = config.vocab_size * config.hidden_size
    params['position_embeddings'] = config.max_position_embeddings * config.hidden_size
    params['token_type_embeddings'] = config.type_vocab_size * config.hidden_size
    
    # Per layer parameters
    per_layer = 0
    
    # Attention (Q, K, V, O projections)
    per_layer += 4 * config.hidden_size * config.hidden_size
    
    # FFN (gate, up, down projections)
    per_layer += 2 * config.hidden_size * config.intermediate_size  # gate + up
    per_layer += config.intermediate_size * config.hidden_size  # down
    
    # Normalization parameters (gamma, beta for 2 norms per layer)
    per_layer += 4 * config.hidden_size
    
    params['per_layer'] = per_layer
    params['all_layers'] = per_layer * config.num_hidden_layers
    
    # MLM head
    params['mlm_head'] = config.hidden_size * config.vocab_size
    
    # Final norm
    params['final_norm'] = 2 * config.hidden_size
    
    # Total
    params['total'] = sum(v for k, v in params.items() if k not in ['per_layer'])
    
    # Account for ternary quantization (roughly 1.58 bits vs 32 bits)
    # This is approximate since not all parameters are ternarized
    ternary_reduction = 1.58 / 32
    params['effective_ternary'] = params['total'] * ternary_reduction
    
    return params


if __name__ == "__main__":
    # Test configurations
    for size in ["small", "base", "large"]:
        config = get_model_config(size)
        params = calculate_model_params(config)
        
        print(f"\n{size.upper()} Model Configuration:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  Intermediate size: {config.intermediate_size}")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Parameters (M): {params['total'] / 1e6:.1f}M")
        print(f"  Effective ternary params (M): {params['effective_ternary'] / 1e6:.1f}M")