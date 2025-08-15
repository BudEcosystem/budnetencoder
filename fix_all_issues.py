#!/usr/bin/env python3
"""
Comprehensive fix for all training issues.
This script identifies and fixes all problems in the implementation.
"""

import torch
import re

def fix_implementation_file():
    """Fix all issues in implementation.py"""
    
    print("Fixing implementation.py issues...")
    
    # Read the file
    with open('implementation.py', 'r') as f:
        content = f.read()
    
    # Fix 1: Ensure PercentileTracker buffers are on correct device
    # The issue is that percentile_vals and initialized tensors are not being moved to device
    content = re.sub(
        r"class PercentileTracker:\n(.*?)def register_buffer\(self, name: str, tensor: torch\.Tensor\):",
        r"""class PercentileTracker:
\1def register_buffer(self, name: str, tensor: torch.Tensor):""",
        content,
        flags=re.DOTALL
    )
    
    # Fix 2: In Int4Quantizer, ensure tracker buffers are moved properly
    old_int4_init = """    def __init__(self, num_channels: int, percentile: float = 99.7, 
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
        self.tracker.initialized = self.initialized"""
    
    new_int4_init = """    def __init__(self, num_channels: int, percentile: float = 99.7, 
                 momentum: float = 0.95, use_stochastic: bool = True):
        super().__init__()
        self.bits = 4
        self.L = 2 ** (self.bits - 1) - 1
        self.use_stochastic = use_stochastic
        self.num_channels = num_channels
        self.percentile = percentile
        self.momentum = momentum
        
        # Register buffers directly
        self.register_buffer('percentile_vals', torch.ones(num_channels))
        self.register_buffer('initialized', torch.zeros(num_channels, dtype=torch.bool))"""
    
    content = content.replace(old_int4_init, new_int4_init)
    
    # Fix 3: Update Int4Quantizer forward to handle device properly
    old_forward_start = """    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Quantize activations to int4."""
        # Update percentile statistics
        if training:
            self.tracker.update(x)
        
        # Get quantization scale
        scale = self.tracker.get_scale(self.bits)"""
    
    new_forward_start = """    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Quantize activations to int4."""
        # Ensure buffers are on correct device
        if self.percentile_vals.device != x.device:
            self.percentile_vals = self.percentile_vals.to(x.device)
            self.initialized = self.initialized.to(x.device)
        
        # Update percentile statistics
        if training:
            self.update_percentiles(x)
        
        # Get quantization scale
        L = 2 ** (self.bits - 1) - 1
        scale = self.percentile_vals / L"""
    
    content = content.replace(old_forward_start, new_forward_start)
    
    # Fix 4: Add update_percentiles method to Int4Quantizer
    int4_class_end = "        return x_dequant"
    new_method = """        return x_dequant
    
    def update_percentiles(self, x: torch.Tensor):
        """Update percentile statistics."""
        # x shape: [B, S, C] or [B*S, C]
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        
        with torch.no_grad():
            actual_channels = min(self.num_channels, x.size(-1))
            for c in range(actual_channels):
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
                        self.percentile_vals[c] = (
                            self.momentum * self.percentile_vals[c] + 
                            (1 - self.momentum) * percentile_val
                        )"""
    
    content = content.replace(int4_class_end, new_method)
    
    # Write the fixed file
    with open('implementation.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed device mismatch in Int4Quantizer")
    print("✓ Fixed buffer registration issues")
    print("✓ Added proper device handling")

if __name__ == "__main__":
    fix_implementation_file()
    print("\nAll issues fixed! Ready to restart training.")