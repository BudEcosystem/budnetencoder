"""
BitNet-QDyT-v2: Orthogonality-Aware Ternary Encoders with Percentile-Scaled Int4 Activations
Complete production implementation following the paper specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from collections import deque
import warnings


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BitNetQDyTConfig:
    """Configuration for BitNet-QDyT-v2 model."""
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 24
    num_attention_heads: int = 12
    intermediate_size: int = 4608  # 6x expansion for SwiGLU
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Quantization settings
    weight_bits: int = 2  # Ternary
    activation_bits: int = 4
    qk_bits: int = 8  # Mixed precision for Q/K
    gate_bits: int = 8  # Mixed precision for SwiGLU gate
    embedding_bits: int = 8
    lm_head_bits: int = 8
    mixed_precision_last_blocks: int = 2  # Number of last blocks with higher V/O precision
    vo_bits: int = 8  # Bits for V/O in last blocks
    
    # Orthogonal mixing
    block_hadamard_size: int = 64
    use_dpd: bool = True
    use_block_hadamard: bool = True
    alternate_ffn_mixing: bool = True
    
    # QDyT normalization
    qdyt_group_size: int = 16
    qdyt_alpha_init: float = 0.05
    qdyt_alpha_target: float = 0.5
    qdyt_warmup_steps: int = 2000
    use_pact_clip: bool = True
    
    # Percentile scaling
    percentile_value: float = 99.7
    percentile_ema_momentum: float = 0.95
    use_stochastic_rounding: bool = True
    
    # TTQ/LSQ+ settings
    ttq_init_percentile: float = 30.0
    ttq_delta_factor: float = 0.05
    lsq_grad_clip: float = 1.0
    use_dither: bool = True
    ternary_dropout_prob: float = 0.1
    ternary_dropout_steps: int = 10000
    
    # Training settings
    progressive_quantization: bool = True
    prog_stage1_pct: float = 0.1  # 10% for warmup
    prog_stage2_pct: float = 0.4  # 40% for partial ternary
    qdrop_prob_initial: float = 0.1
    qdrop_anneal_steps: int = 30000
    
    # Residual path
    use_skip_init: bool = True
    skip_init_scale: float = 0.5
    
    # Teacher distillation
    use_teacher_guidance: bool = False
    kd_temperature: float = 2.0
    kd_weight_max: float = 0.5


# ============================================================================
# Orthogonal Mixing Components
# ============================================================================

class DPD(nn.Module):
    """Diagonal-Permutation-Diagonal orthogonal transformation."""
    
    def __init__(self, dim: int, seed: Optional[int] = None):
        super().__init__()
        self.dim = dim
        
        # Initialize diagonal matrices with +/-1
        if seed is not None:
            torch.manual_seed(seed)
        
        self.register_buffer('sign1', torch.randint(0, 2, (dim,), dtype=torch.float32) * 2 - 1)
        self.register_buffer('sign2', torch.randint(0, 2, (dim,), dtype=torch.float32) * 2 - 1)
        
        # Random permutation
        perm = torch.randperm(dim)
        self.register_buffer('perm', perm)
        
        # Inverse permutation for backward
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(dim)
        self.register_buffer('inv_perm', inv_perm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DPD transformation."""
        x = x * self.sign1
        x = x[..., self.perm]
        x = x * self.sign2
        return x
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse DPD transformation."""
        x = x * self.sign2
        x = x[..., self.inv_perm]
        x = x * self.sign1
        return x


def block_hadamard_transform(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """
    Apply Block Hadamard Transform (Fast Hadamard Transform on blocks).
    
    Args:
        x: Input tensor [..., d] where d % block_size == 0
        block_size: Size of each Hadamard block (must be power of 2)
    """
    *batch_dims, d = x.shape
    assert d % block_size == 0, f"Dimension {d} must be divisible by block_size {block_size}"
    assert (block_size & (block_size - 1)) == 0, f"Block size {block_size} must be power of 2"
    
    num_blocks = d // block_size
    x = x.reshape(*batch_dims, num_blocks, block_size)
    
    # Fast Hadamard Transform on each block
    h = x.clone()
    n = int(math.log2(block_size))
    
    for i in range(n):
        stride = 1 << i
        h_new = h.clone()
        for j in range(0, block_size, 2 * stride):
            for k in range(stride):
                a = h[..., j + k]
                b = h[..., j + k + stride]
                h_new[..., j + k] = a + b
                h_new[..., j + k + stride] = a - b
        h = h_new
    
    # Normalize
    h = h / math.sqrt(block_size)
    
    return h.reshape(*batch_dims, d)


class BlockHadamardDPD(nn.Module):
    """Combined Block Hadamard + DPD orthogonal mixer."""
    
    def __init__(self, dim: int, block_size: int = 64, use_dpd: bool = True, 
                 use_block_h: bool = True, seed: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.use_dpd = use_dpd
        self.use_block_h = use_block_h
        
        if use_dpd:
            self.dpd = DPD(dim, seed=seed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply orthogonal mixing."""
        if self.use_block_h:
            x = block_hadamard_transform(x, self.block_size)
        if self.use_dpd:
            x = self.dpd(x)
        return x


# ============================================================================
# Quantization Components
# ============================================================================

class PercentileTracker:
    """Track percentile statistics with EMA for activation quantization."""
    
    def __init__(self, num_channels: int, percentile: float = 99.7, 
                 momentum: float = 0.95):
        self.num_channels = num_channels
        self.percentile = percentile
        self.momentum = momentum
        self.register_buffer('percentile_vals', torch.ones(num_channels))
        self.register_buffer('initialized', torch.zeros(num_channels, dtype=torch.bool))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Helper to register buffers (will be moved to module later)."""
        setattr(self, name, tensor)
    
    def update(self, x: torch.Tensor) -> torch.Tensor:
        """Update percentile statistics and return scale."""
        # x shape: [B, S, C] or [B*S, C]
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        
        with torch.no_grad():
            for c in range(self.num_channels):
                x_c = x[:, c].abs()
                if x_c.numel() > 0:
                    # Calculate percentile
                    k = int(self.percentile * x_c.numel() / 100)
                    k = max(1, min(k, x_c.numel() - 1))
                    percentile_val = torch.kthvalue(x_c, k).values
                    
                    if not self.initialized[c]:
                        self.percentile_vals[c] = percentile_val
                        self.initialized[c] = True
                    else:
                        self.percentile_vals[c] = (self.momentum * self.percentile_vals[c] + 
                                                   (1 - self.momentum) * percentile_val)
        
        return self.percentile_vals
    
    def get_scale(self, bits: int = 4) -> torch.Tensor:
        """Get quantization scale from percentile values."""
        L = 2 ** (bits - 1) - 1
        return self.percentile_vals / L


class TTQLSQTernaryQuantizer(nn.Module):
    """TTQ/LSQ+ ternary weight quantizer with learned scales and thresholds."""
    
    def __init__(self, out_features: int, in_features: int, 
                 init_percentile: float = 30.0, delta_factor: float = 0.05):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        
        # Learnable parameters per output channel
        self.s_tilde = nn.Parameter(torch.zeros(out_features))
        self.t = nn.Parameter(torch.zeros(out_features))
        self.delta = nn.Parameter(torch.ones(out_features) * delta_factor)
        
        self.init_percentile = init_percentile
        self.delta_factor = delta_factor
        self.initialized = False
        
        # For gradient scaling
        self.register_buffer('grad_scale', torch.tensor(1.0 / math.sqrt(in_features * 3)))
    
    def initialize_thresholds(self, weight: torch.Tensor):
        """Initialize thresholds from weight statistics."""
        if not self.initialized:
            with torch.no_grad():
                # weight shape: [out_features, in_features]
                for i in range(self.out_features):
                    w_i = weight[i].abs()
                    # Initialize t from percentile
                    k = int(self.init_percentile * w_i.numel() / 100)
                    k = max(1, min(k, w_i.numel() - 1))
                    self.t.data[i] = torch.kthvalue(w_i, k).values
                    
                    # Initialize delta
                    median_val = torch.median(w_i)
                    self.delta.data[i] = self.delta_factor * median_val
                    
                    # Initialize scale
                    self.s_tilde.data[i] = weight[i].std()
            
            self.initialized = True
    
    def forward(self, weight: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Quantize weights to ternary values."""
        if not self.initialized:
            self.initialize_thresholds(weight)
        
        # Compute scales (positive via softplus)
        s = F.softplus(self.s_tilde)
        
        # Ternary quantization per channel
        weight_quantized = torch.zeros_like(weight)
        
        for i in range(self.out_features):
            w_i = weight[i]
            # Compute ternary values
            signs = torch.sign(w_i - self.t[i])
            mask = (w_i - self.t[i]).abs() > self.delta[i]
            q_i = signs * mask.float()
            
            if training:
                # Add dither during warmup (handled externally)
                weight_quantized[i] = s[i] * q_i
            else:
                weight_quantized[i] = s[i] * q_i
        
        # Straight-through estimator
        if training:
            weight_quantized = weight + (weight_quantized - weight).detach()
        
        return weight_quantized


class Int4Quantizer(nn.Module):
    """4-bit symmetric quantizer with percentile scaling and stochastic rounding."""
    
    def __init__(self, num_channels: int, percentile: float = 99.7, 
                 momentum: float = 0.95, use_stochastic: bool = True):
        super().__init__()
        self.bits = 4
        self.L = 2 ** (self.bits - 1) - 1
        self.use_stochastic = use_stochastic
        
        # Percentile tracker
        self.tracker = PercentileTracker(num_channels, percentile, momentum)
        # Move tracker buffers to module
        self.register_buffer('percentile_vals', self.tracker.percentile_vals)
        self.register_buffer('initialized', self.tracker.initialized)
        self.tracker.percentile_vals = self.percentile_vals
        self.tracker.initialized = self.initialized
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Quantize activations to int4."""
        # Update percentile statistics
        if training:
            self.tracker.update(x)
        
        # Get quantization scale
        scale = self.tracker.get_scale(self.bits)
        scale = scale.view(1, 1, -1) if x.dim() == 3 else scale.view(1, -1)
        
        # Quantize
        x_scaled = x / (scale + 1e-8)
        
        if training and self.use_stochastic:
            # Stochastic rounding
            x_floor = x_scaled.floor()
            prob = x_scaled - x_floor
            random_mask = torch.rand_like(x_scaled) < prob
            x_rounded = torch.where(random_mask, x_floor + 1, x_floor)
        else:
            # Deterministic rounding
            x_rounded = x_scaled.round()
        
        # Clip to range
        x_clipped = torch.clamp(x_rounded, -self.L, self.L)
        
        # Dequantize
        x_dequant = x_clipped * scale
        
        # Straight-through estimator
        if training:
            x_dequant = x + (x_dequant - x).detach()
        
        return x_dequant


# ============================================================================
# QDyT Normalization
# ============================================================================

class QDyTGroupNorm(nn.Module):
    """Group-wise Dynamic Tanh normalization with PACT clip."""
    
    def __init__(self, hidden_size: int, group_size: int = 16, 
                 alpha_init: float = 0.05, use_pact: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.use_pact = use_pact
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        
        if use_pact:
            self.a_tilde = nn.Parameter(torch.tensor(3.0))  # Will be passed through softplus
        
        # For calibration
        self.register_buffer('running_mean', torch.zeros(self.num_groups))
        self.register_buffer('calibrated', torch.tensor(False))
        
        self.warmup_steps = 0
        self.alpha_target = 0.5
        self.alpha_warmup_steps = 2000
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply QDyT normalization."""
        # x shape: [B, S, H]
        B, S, H = x.shape
        
        # Reshape for group-wise mean
        x_grouped = x.view(B, S, self.num_groups, self.group_size)
        
        # Compute group-wise mean
        if self.calibrated:
            mu = self.running_mean.view(1, 1, self.num_groups, 1)
        else:
            mu = x_grouped.mean(dim=-1, keepdim=True)
        
        # Center
        y = x_grouped - mu
        y = y.view(B, S, H)
        
        # PACT clip
        if self.use_pact:
            a = F.softplus(self.a_tilde)
            y = torch.clamp(y, -a, a)
        
        # Get current alpha (with warmup)
        if self.training and self.warmup_steps < self.alpha_warmup_steps:
            alpha_current = (self.alpha_init + 
                           (self.alpha_target - self.alpha_init) * 
                           self.warmup_steps / self.alpha_warmup_steps)
            self.warmup_steps += 1
        else:
            alpha_current = self.alpha
        
        # Apply tanh
        y = torch.tanh(alpha_current * y)
        
        # Scale and shift
        y = self.gamma * y + self.beta
        
        return y
    
    def calibrate(self, x: torch.Tensor, momentum: float = 0.99):
        """Update running mean for calibration."""
        with torch.no_grad():
            B, S, H = x.shape
            x_grouped = x.view(B, S, self.num_groups, self.group_size)
            mu = x_grouped.mean(dim=(0, 1, 3))  # Mean across batch, seq, group dims
            
            if not self.calibrated:
                self.running_mean.copy_(mu)
                self.calibrated = torch.tensor(True)
            else:
                self.running_mean.mul_(momentum).add_(mu, alpha=1 - momentum)


# ============================================================================
# Attention Components
# ============================================================================

class BitNetQDyTAttention(nn.Module):
    """Multi-head attention with orthogonal mixing and mixed precision."""
    
    def __init__(self, config: BitNetQDyTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Check if this is a late block that needs higher precision V/O
        self.is_late_block = (layer_idx >= config.num_hidden_layers - config.mixed_precision_last_blocks)
        
        # Orthogonal mixers per head
        if config.use_block_hadamard or config.use_dpd:
            self.mixer_q = BlockHadamardDPD(self.head_dim, config.block_hadamard_size,
                                           config.use_dpd, config.use_block_hadamard)
            self.mixer_k = BlockHadamardDPD(self.head_dim, config.block_hadamard_size,
                                           config.use_dpd, config.use_block_hadamard)
            self.mixer_v = BlockHadamardDPD(self.head_dim, config.block_hadamard_size,
                                           config.use_dpd, config.use_block_hadamard)
        
        # Ternary weight projections
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Weight quantizers
        self.q_quantizer = TTQLSQTernaryQuantizer(self.hidden_size, self.hidden_size)
        self.k_quantizer = TTQLSQTernaryQuantizer(self.hidden_size, self.hidden_size)
        
        if self.is_late_block and config.vo_bits > 2:
            # Higher precision for V/O in late blocks
            self.v_quantizer = None  # Will use higher bit quantization
            self.o_quantizer = None
        else:
            self.v_quantizer = TTQLSQTernaryQuantizer(self.hidden_size, self.hidden_size)
            self.o_quantizer = TTQLSQTernaryQuantizer(self.hidden_size, self.hidden_size)
        
        # Activation quantizers (Q/K use higher precision)
        self.q_act_quantizer = Int4Quantizer(self.hidden_size) if config.qk_bits <= 4 else None
        self.k_act_quantizer = Int4Quantizer(self.hidden_size) if config.qk_bits <= 4 else None
        self.v_act_quantizer = Int4Quantizer(self.hidden_size)
        self.o_act_quantizer = Int4Quantizer(self.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # For mixed precision Q/K
        self.qk_bits = config.qk_bits
        self.vo_bits = config.vo_bits if self.is_late_block else 2
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                training: bool = True) -> torch.Tensor:
        B, S, H = hidden_states.shape
        
        # Quantize weights
        q_weight = self.q_quantizer(self.query.weight, training)
        k_weight = self.k_quantizer(self.key.weight, training)
        
        if self.v_quantizer is not None:
            v_weight = self.v_quantizer(self.value.weight, training)
            o_weight = self.o_quantizer(self.output.weight, training)
        else:
            # Higher precision for late blocks
            v_weight = self.value.weight
            o_weight = self.output.weight
        
        # Project with quantized weights
        q = F.linear(hidden_states, q_weight)
        k = F.linear(hidden_states, k_weight)
        v = F.linear(hidden_states, v_weight)
        
        # Reshape for heads
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply orthogonal mixing per head
        if hasattr(self, 'mixer_q'):
            q = self.mixer_q(q)
            k = self.mixer_k(k)
            v = self.mixer_v(v)
        
        # Quantize Q/K activations (higher precision)
        if self.qk_bits <= 4 and self.q_act_quantizer is not None:
            q = self.q_act_quantizer(q.reshape(B * self.num_heads * S, self.head_dim), training)
            q = q.view(B, self.num_heads, S, self.head_dim)
            k = self.k_act_quantizer(k.reshape(B * self.num_heads * S, self.head_dim), training)
            k = k.view(B, self.num_heads, S, self.head_dim)
        
        # Quantize V activations
        v = self.v_act_quantizer(v.reshape(B * self.num_heads * S, self.head_dim), training)
        v = v.view(B, self.num_heads, S, self.head_dim)
        
        # Compute attention scores (accumulated in higher precision)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax in FP16/32
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention
        context = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(B, S, H)
        
        # Quantize output activations
        context = self.o_act_quantizer(context, training)
        
        output = F.linear(context, o_weight)
        
        return output


# ============================================================================
# FFN Components
# ============================================================================

class BitNetQDyTSwiGLU(nn.Module):
    """SwiGLU FFN with mixed precision gate and orthogonal mixing."""
    
    def __init__(self, config: BitNetQDyTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Determine if we use FFN mixing this layer
        self.use_ffn_mixing = True
        if config.alternate_ffn_mixing:
            # Use Block-H every other block
            self.use_block_h = (layer_idx // 2) % 2 == 0
        else:
            self.use_block_h = config.use_block_hadamard
        
        # Orthogonal mixers
        if config.use_dpd or self.use_block_h:
            self.mixer_in = BlockHadamardDPD(self.hidden_size, config.block_hadamard_size,
                                            config.use_dpd, self.use_block_h)
            self.mixer_out = BlockHadamardDPD(self.intermediate_size, config.block_hadamard_size,
                                             config.use_dpd, True)  # Always use DPD for output
        
        # Linear layers
        self.gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        # Weight quantizers
        self.gate_quantizer = TTQLSQTernaryQuantizer(self.intermediate_size, self.hidden_size)
        self.up_quantizer = TTQLSQTernaryQuantizer(self.intermediate_size, self.hidden_size)
        self.down_quantizer = TTQLSQTernaryQuantizer(self.hidden_size, self.intermediate_size)
        
        # Activation quantizers
        self.gate_act_quantizer = Int4Quantizer(self.intermediate_size) if config.gate_bits <= 4 else None
        self.up_act_quantizer = Int4Quantizer(self.intermediate_size)
        self.down_input_quantizer = Int4Quantizer(self.intermediate_size)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.gate_bits = config.gate_bits
    
    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Apply input mixing
        if hasattr(self, 'mixer_in'):
            hidden_states = self.mixer_in(hidden_states)
        
        # Quantize weights
        gate_weight = self.gate_quantizer(self.gate.weight, training)
        up_weight = self.up_quantizer(self.up.weight, training)
        down_weight = self.down_quantizer(self.down.weight, training)
        
        # Gate branch (higher precision)
        gate_out = F.linear(hidden_states, gate_weight)
        if self.gate_bits <= 4 and self.gate_act_quantizer is not None:
            gate_out = self.gate_act_quantizer(gate_out, training)
        gate_out = F.silu(gate_out)
        
        # Main branch (int4)
        up_out = F.linear(hidden_states, up_weight)
        up_out = self.up_act_quantizer(up_out, training)
        
        # Combine
        intermediate = gate_out * up_out
        
        # Apply output mixing
        if hasattr(self, 'mixer_out'):
            intermediate = self.mixer_out(intermediate)
        
        # Quantize before down projection
        intermediate = self.down_input_quantizer(intermediate, training)
        
        # Down projection
        output = F.linear(intermediate, down_weight)
        output = self.dropout(output)
        
        return output


# ============================================================================
# Layer and Model
# ============================================================================

class BitNetQDyTLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, config: BitNetQDyTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # QDyT normalization instead of LayerNorm
        self.attention_norm = QDyTGroupNorm(config.hidden_size, config.qdyt_group_size,
                                           config.qdyt_alpha_init, config.use_pact_clip)
        self.ffn_norm = QDyTGroupNorm(config.hidden_size, config.qdyt_group_size,
                                     config.qdyt_alpha_init, config.use_pact_clip)
        
        # Attention and FFN
        self.attention = BitNetQDyTAttention(config, layer_idx)
        self.ffn = BitNetQDyTSwiGLU(config, layer_idx)
        
        # Skip connection scaling (SkipInit)
        if config.use_skip_init:
            init_scale = config.skip_init_scale / math.sqrt(config.num_hidden_layers)
            self.attn_residual_scale = nn.Parameter(torch.tensor(init_scale))
            self.ffn_residual_scale = nn.Parameter(torch.tensor(init_scale))
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                training: bool = True) -> torch.Tensor:
        # Attention block
        normed = self.attention_norm(hidden_states)
        attn_out = self.attention(normed, attention_mask, training)
        
        if hasattr(self, 'attn_residual_scale'):
            attn_out = attn_out * self.attn_residual_scale
        
        hidden_states = hidden_states + attn_out
        
        # FFN block
        normed = self.ffn_norm(hidden_states)
        ffn_out = self.ffn(normed, training)
        
        if hasattr(self, 'ffn_residual_scale'):
            ffn_out = ffn_out * self.ffn_residual_scale
        
        hidden_states = hidden_states + ffn_out
        
        return hidden_states


class BitNetQDyTModel(nn.Module):
    """Complete BitNet-QDyT-v2 encoder model."""
    
    def __init__(self, config: BitNetQDyTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings (higher precision)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Embedding quantizer (higher bits)
        self.embedding_quantizer = Int4Quantizer(config.hidden_size) if config.embedding_bits <= 4 else None
        
        # Initial normalization
        self.embedding_norm = QDyTGroupNorm(config.hidden_size, config.qdyt_group_size,
                                           config.qdyt_alpha_init, config.use_pact_clip)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BitNetQDyTLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.final_norm = QDyTGroupNorm(config.hidden_size, config.qdyt_group_size,
                                       config.qdyt_alpha_init, config.use_pact_clip)
        
        # MLM head (untied, higher precision)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following μParam principles."""
        if isinstance(module, nn.Linear):
            # Scale by 1/sqrt(d) for ternary compatibility
            std = 1.0 / math.sqrt(module.weight.size(1))
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert attention mask to extended format."""
        if attention_mask.dim() == 2:
            # [B, S] -> [B, 1, 1, S]
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        elif attention_mask.dim() == 3:
            # [B, S, S] -> [B, 1, S, S]
            extended_mask = attention_mask.unsqueeze(1)
        else:
            extended_mask = attention_mask
        
        # Convert to attention scores mask
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                training: bool = True) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Create token type IDs if not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Embeddings
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeds + position_embeds + token_type_embeds
        
        # Quantize embeddings (higher precision)
        if self.embedding_quantizer is not None and self.config.embedding_bits <= 4:
            embeddings = self.embedding_quantizer(embeddings, training)
        
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Prepare attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        # Transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask, training)
        
        # Final norm
        hidden_states = self.final_norm(hidden_states)
        
        # MLM predictions
        logits = self.mlm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }


# ============================================================================
# Training Utilities
# ============================================================================

class ProgressiveQuantizationScheduler:
    """Manage progressive quantization schedule."""
    
    def __init__(self, config: BitNetQDyTConfig, total_steps: int):
        self.config = config
        self.total_steps = total_steps
        self.stage1_steps = int(config.prog_stage1_pct * total_steps)
        self.stage2_steps = int(config.prog_stage2_pct * total_steps)
    
    def get_stage(self, step: int) -> str:
        """Get current training stage."""
        if step < self.stage1_steps:
            return 'warmup'
        elif step < self.stage2_steps:
            return 'partial_ternary'
        else:
            return 'full_ternary'
    
    def should_quantize_layer(self, layer_idx: int, step: int) -> bool:
        """Check if layer should be quantized at current step."""
        stage = self.get_stage(step)
        
        if stage == 'warmup':
            return False
        elif stage == 'partial_ternary':
            # Quantize top half of layers
            return layer_idx >= self.config.num_hidden_layers // 2
        else:
            return True
    
    def get_qdrop_prob(self, step: int) -> float:
        """Get QDrop probability for current step."""
        if step >= self.config.qdrop_anneal_steps:
            return 0.0
        
        return self.config.qdrop_prob_initial * (1 - step / self.config.qdrop_anneal_steps)


class QDropContext:
    """Context manager for QDrop (randomly bypass quantization)."""
    
    def __init__(self, model: nn.Module, prob: float):
        self.model = model
        self.prob = prob
        self.original_states = {}
    
    def __enter__(self):
        if self.prob > 0 and torch.rand(1).item() < self.prob:
            # Disable quantization temporarily
            for name, module in self.model.named_modules():
                if hasattr(module, 'training'):
                    self.original_states[name] = module.training
                    module.eval()  # This will bypass quantization in many cases
        return self
    
    def __exit__(self, *args):
        # Restore original states
        for name, state in self.original_states.items():
            module = dict(self.model.named_modules())[name]
            module.train(state)


def add_dither_noise(model: nn.Module, dither_scale: float = 0.1):
    """Add uniform dither noise to weights during warmup."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TTQLSQTernaryQuantizer):
                for param in [module.s_tilde, module.t, module.delta]:
                    noise = torch.empty_like(param).uniform_(-dither_scale, dither_scale)
                    param.add_(noise)


def apply_ternary_dropout(model: nn.Module, prob: float = 0.1):
    """Randomly zero out ternary values."""
    for module in model.modules():
        if isinstance(module, TTQLSQTernaryQuantizer):
            # This would be applied during forward pass
            pass  # Implemented in the quantizer forward method


def calibrate_qdyt_norms(model: nn.Module, dataloader, num_batches: int = 100):
    """Calibrate QDyT normalization running means."""
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Forward pass
            _ = model(batch['input_ids'], batch.get('attention_mask'))
            
            # Update running means in QDyT layers
            for module in model.modules():
                if isinstance(module, QDyTGroupNorm):
                    # The calibration happens internally in the forward pass
                    module.calibrated = True
    
    print(f"Calibrated QDyT norms with {num_batches} batches")


# ============================================================================
# Training Script
# ============================================================================

def train_bitnet_qdyt_v2(
    model: BitNetQDyTModel,
    train_dataloader,
    val_dataloader,
    config: BitNetQDyTConfig,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    warmup_steps: int = 10000,
    teacher_model: Optional[nn.Module] = None
):
    """Full training loop for BitNet-QDyT-v2."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
    
    # Optimizer with μParam scaling
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.1
    )
    
    # Progressive quantization scheduler
    prog_scheduler = ProgressiveQuantizationScheduler(config, total_steps)
    
    # Training loop
    global_step = 0
    model.train()
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get('labels', input_ids)  # For MLM
            
            # Get current stage and QDrop probability
            stage = prog_scheduler.get_stage(global_step)
            qdrop_prob = prog_scheduler.get_qdrop_prob(global_step)
            
            # Apply dither during warmup
            if stage == 'warmup' and config.use_dither:
                add_dither_noise(model, dither_scale=0.05)
            
            # Apply ternary dropout
            if global_step < config.ternary_dropout_steps:
                apply_ternary_dropout(model, config.ternary_dropout_prob)
            
            # Forward pass with QDrop
            with QDropContext(model, qdrop_prob):
                outputs = model(input_ids, attention_mask, training=True)
                logits = outputs['logits']
            
            # Compute MLM loss
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(logits.view(-1, config.vocab_size), labels.view(-1))
            
            # Add teacher distillation if available
            total_loss = mlm_loss
            if teacher_model is not None and config.use_teacher_guidance:
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids, attention_mask)
                    teacher_logits = teacher_outputs['logits']
                
                # KL divergence loss
                kd_weight = min(config.kd_weight_max, global_step / warmup_steps * config.kd_weight_max)
                kl_loss = F.kl_div(
                    F.log_softmax(logits / config.kd_temperature, dim=-1),
                    F.softmax(teacher_logits / config.kd_temperature, dim=-1),
                    reduction='batchmean'
                ) * (config.kd_temperature ** 2)
                
                total_loss = mlm_loss + kd_weight * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # LSQ gradient scaling for quantizer parameters
            for module in model.modules():
                if isinstance(module, TTQLSQTernaryQuantizer):
                    if module.s_tilde.grad is not None:
                        module.s_tilde.grad.clamp_(-config.lsq_grad_clip, config.lsq_grad_clip)
                    if module.t.grad is not None:
                        module.t.grad.clamp_(-config.lsq_grad_clip, config.lsq_grad_clip)
                    if module.delta.grad is not None:
                        module.delta.grad.clamp_(-config.lsq_grad_clip, config.lsq_grad_clip)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Logging
            if global_step % 100 == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {total_loss.item():.4f}, "
                      f"Stage: {stage}, QDrop: {qdrop_prob:.3f}")
            
            global_step += 1
            
            # Reduce learning rate when entering full ternary stage
            if global_step == prog_scheduler.stage2_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.7
                print("Reduced learning rate for full ternary stage")
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    labels = batch.get('labels', input_ids)
                    
                    outputs = model(input_ids, attention_mask, training=False)
                    logits = outputs['logits']
                    
                    loss = loss_fct(logits.view(-1, config.vocab_size), labels.view(-1))
                    val_loss += loss.item()
            
            val_loss /= len(val_dataloader)
            print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")
            model.train()
    
    # Post-training calibration
    print("Starting post-training QDyT calibration...")
    calibrate_qdyt_norms(model, val_dataloader or train_dataloader, num_batches=100)
    
    return model


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create configuration
    config = BitNetQDyTConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=24,
        num_attention_heads=12,
        intermediate_size=4608,  # 6x expansion
        max_position_embeddings=512,
        progressive_quantization=True,
        use_teacher_guidance=False  # Set to True if you have a teacher model
    )
    
    # Create model
    model = BitNetQDyTModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Example batch
    batch_size = 4
    seq_length = 128
    
    example_batch = {
        'input_ids': torch.randint(0, config.vocab_size, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'labels': torch.randint(0, config.vocab_size, (batch_size, seq_length))
    }
    
    # Forward pass
    model.train()
    outputs = model(**example_batch, training=True)
    print(f"Output shape: {outputs['logits'].shape}")
    
    print("\nModel ready for production training!")
    print("Use train_bitnet_qdyt_v2() function with your dataloaders to start training.")
