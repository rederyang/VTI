"""
Competitor intervention methods for vision feature manipulation.
Uses PyTorch forward hooks to intervene on Vision Encoder output.
"""

import torch
import torch.nn.functional as F


class CompetitorIntervention:
    """Vision feature intervention via forward hook.
    
    Registers a hook on CLIPVisionTower to modify output features
    before they are passed to the projector.
    
    Args:
        method: One of 'clipping', 'smoothing', 'quantization'
        **kwargs: Method-specific parameters
    """
    
    def __init__(self, method: str, **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.handle = None
    
    def hook_fn(self, module, inputs, output):
        """Forward hook function applied to vision tower output.
        
        Args:
            module: The hooked module (CLIPVisionTower)
            inputs: Tuple of inputs to the module
            output: Output tensor [B, 576, D] for LLaVA-1.5
        
        Returns:
            Modified output tensor with same shape
        """
        # CLIPVisionTower.forward() returns tensor [B, 576, D], not tuple
        if self.method == 'clipping':
            return self.dynamic_clipping(output, **self.kwargs)
        elif self.method == 'smoothing':
            return self.spatial_smoothing(output, **self.kwargs)
        elif self.method == 'quantization':
            return self.feature_quantization(output, **self.kwargs)
        return output
    
    def register(self, module):
        """Register the hook on the module.
        
        Args:
            module: The module to register the hook on
        """
        self.handle = module.register_forward_hook(self.hook_fn)
        print(f"Registered {self.method} intervention on {module.__class__.__name__}")
    
    def remove(self):
        """Remove the registered hook."""
        if self.handle:
            self.handle.remove()
            self.handle = None
            print(f"Removed {self.method} intervention")

    # --- Method Implementations ---
    
    @staticmethod
    def dynamic_clipping(x, percentile=95, mode="per-channel"):
        """Clip outlier feature values based on percentile.
        
        Addresses the ~15% unstable features mentioned in VTI paper
        by truncating extreme values to the specified percentile range.
        
        Args:
            x: Input tensor [B, L, D]
            percentile: Upper percentile for clipping (e.g., 95 means clip to 5-95%)
        
        Returns:
            Clipped tensor with same shape
        """
        # handle the case where x is a tuple
        is_tuple = False
        if isinstance(x, tuple):
            is_tuple = True
            x = x[0]

        # handle the case where x is not float32
        x_dtype = x.dtype
        if x_dtype != torch.float32:
            x = x.float()

        # Compute bounds along the dimension specified by mode
        if mode == "per-channel":  # every channel is clipped independently
            lower = torch.quantile(x, (100 - percentile) / 100.0, dim=1, keepdim=True)
            upper = torch.quantile(x, percentile / 100.0, dim=1, keepdim=True)
        elif mode == "per-token":  # every token is clipped independently
            lower = torch.quantile(x, (100 - percentile) / 100.0, dim=2, keepdim=True)
            upper = torch.quantile(x, percentile / 100.0, dim=2, keepdim=True)
        elif mode == "global":  # all tokens, all channels are clipped together
            B, L, D = x.shape
            x_cluster = x.reshape(B, -1)
            lower = torch.quantile(x_cluster, (100 - percentile) / 100.0, dim=1, keepdim=True)
            upper = torch.quantile(x_cluster, percentile / 100.0, dim=1, keepdim=True)
            lower = lower.reshape(B, 1, 1)
            upper = upper.reshape(B, 1, 1)
        else:
            raise ValueError(f"Invalid clipping mode: {mode}")

        assert (upper > lower).all(), "Upper bound must be greater than lower bound"

        x_clamped = torch.clamp(x, min=lower, max=upper)

        # Count the number of entries that are different after clamping
        different_entries = (x_clamped != x).sum()
        print(f"Clipping {different_entries} entries out of {x.numel()}")

        if x_dtype != torch.float32:
            x_clamped = x_clamped.to(x_dtype)

        if is_tuple:
            x_clamped = (x_clamped,)

        return x_clamped

    @staticmethod
    def spatial_smoothing(x, kernel_size=3, grid_size=24):
        """Apply spatial average pooling to smooth features.
        
        Treats the sequence as a 2D spatial grid and applies
        local averaging to suppress high-frequency noise.
        
        Args:
            x: Input tensor [B, L, D], L should be grid_size^2
            kernel_size: Size of the averaging kernel (odd number)
            grid_size: Spatial grid size (24 for LLaVA-1.5 with 336px input)
        
        Returns:
            Smoothed tensor with same shape
        """
        B, L, D = x.shape
        if L != grid_size * grid_size:
            # Skip if shape doesn't match expected grid
            print("Warning: Spatial smoothing input shape does not match expected grid size")
            return x
        
        # Reshape to 2D spatial format: [B, L, D] -> [B, D, H, W]
        x_2d = x.view(B, grid_size, grid_size, D).permute(0, 3, 1, 2)
        
        # Apply average pooling with same-size output
        padding = kernel_size // 2
        x_smooth = F.avg_pool2d(
            x_2d, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding, 
            count_include_pad=False
        )
        
        # Reshape back: [B, D, H, W] -> [B, L, D]
        return x_smooth.permute(0, 2, 3, 1).view(B, L, D)
    
    @staticmethod
    def feature_quantization(x, scale=10.0):
        """Quantize features to reduce precision.
        
        Simulates lower bit-depth by rounding to discrete levels.
        This filters out small perturbations that may cause instability.
        
        Args:
            x: Input tensor [B, L, D]
            scale: Quantization scale (higher = finer granularity)
        
        Returns:
            Quantized tensor with same shape
        """
        return torch.round(x * scale) / scale

