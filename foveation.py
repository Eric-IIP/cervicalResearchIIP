import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FoveatedSamplingConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=11, input_height=256, input_width=256):
        """
        Concrete Foveated Sampling Convolution Layer
        Pre-computes foveated kernels for every position in the feature map
        
        Args:
            in_channels: Input channels (3 from your 1x1 conv)
            out_channels: Output feature channels
            kernel_size: Size of the foveated kernel (should be odd, e.g., 11x11)
            input_height: Height of input feature map
            input_width: Width of input feature map
        """
        super(FoveatedSamplingConv2d, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_height = input_height
        self.input_width = input_width
        self.center = kernel_size // 2
        
        # Define the sampling scales for each ring
        self.scales = {
            'center': 1,      # Direct sampling (red region)
            'ring1': 3,       # 9x area -> ~3x scale (orange region)
            'ring2': 5,       # 25x area -> ~5x scale (purple region)  
            'ring3': 7        # 49x area -> ~7x scale (light blue region)
        }
        
        # Learnable convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Pre-compute all foveated sampling positions for each (y,x) in feature map
        self.register_buffer('foveated_kernels', self._create_all_foveated_kernels())
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_ring_type(self, i, j):
        """Determine which sampling ring a kernel position belongs to"""
        center = self.center
        dist = max(abs(i - center), abs(j - center))
        
        if dist <= 1:
            return 'center'  # 3x3 center region (dist 0 and 1)
        elif dist == 2:
            return 'ring1' 
        elif dist == 3:
            return 'ring2'
        else:
            return 'ring3'
    
    def _create_all_foveated_kernels(self):
        """
        Pre-compute foveated sampling positions for every (y,x) in the feature map
        Returns tensor of shape [H, W, kernel_size, kernel_size, 2] containing (y,x) sampling coordinates
        """
        H, W = self.input_height, self.input_width
        kernels = torch.zeros(H, W, self.kernel_size, self.kernel_size, 2)
        
        for y in range(H):
            for x in range(W):
                for ki in range(self.kernel_size):
                    for kj in range(self.kernel_size):
                        # Determine which ring this kernel position belongs to
                        ring_type = self._get_ring_type(ki, kj)
                        scale = self.scales[ring_type]
                        
                        # Calculate the sampling position
                        # Offset from kernel center, scaled by ring
                        offset_y = (ki - self.center) * scale
                        offset_x = (kj - self.center) * scale
                        
                        # Final sampling position (clamped to image boundaries)
                        sample_y = max(0, min(H-1, y + offset_y))
                        sample_x = max(0, min(W-1, x + offset_x))
                        
                        kernels[y, x, ki, kj, 0] = sample_y
                        kernels[y, x, ki, kj, 1] = sample_x
        
        return kernels
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _sample_at_positions(self, x, sample_coords):
        """
        Sample from input feature map at specified coordinates using bilinear interpolation
        
        Args:
            x: Input tensor [B, C, H, W]
            sample_coords: Sampling coordinates [H, W, kernel_size, kernel_size, 2]
        
        Returns:
            sampled: [B, C, H, W, kernel_size, kernel_size]
        """
        B, C, H, W = x.shape
        
        # Flatten coordinates for grid_sample
        coords_flat = sample_coords.view(-1, 2)  # [H*W*K*K, 2]
        
        # Normalize coordinates to [-1, 1] for grid_sample
        coords_normalized = torch.zeros_like(coords_flat)
        coords_normalized[:, 0] = 2.0 * coords_flat[:, 1] / (W - 1) - 1.0  # x coordinates (note: x comes first in grid_sample)
        coords_normalized[:, 1] = 2.0 * coords_flat[:, 0] / (H - 1) - 1.0  # y coordinates
        
        # Reshape for grid_sample: [B, H*W*K*K, 1, 2]
        grid = coords_normalized.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
        
        # Sample features
        sampled_flat = F.grid_sample(
            x, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )  # [B, C, H*W*K*K, 1]
        
        # Reshape back to [B, C, H, W, K, K]
        sampled = sampled_flat.squeeze(-1).view(B, C, H, W, self.kernel_size, self.kernel_size)
        
        return sampled
    
    def forward(self, x):
        """
        Forward pass of concrete foveated sampling convolution
        
        Args:
            x: Input feature map [B, 3, H, W]
        
        Returns:
            output: [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        
        # Ensure input dimensions match what we pre-computed for
        assert H == self.input_height and W == self.input_width, \
            f"Input size ({H}, {W}) doesn't match expected ({self.input_height}, {self.input_width})"
        
        # Sample features at all pre-computed foveated positions
        # sampled_features: [B, C, H, W, kernel_size, kernel_size]
        sampled_features = self._sample_at_positions(x, self.foveated_kernels)
        
        # Apply convolution: multiply sampled features with learned weights
        # sampled_features: [B, C, H, W, K, K]
        # weight: [out_channels, in_channels, K, K]
        # Use einsum for efficient batched convolution
        output = torch.einsum('bchwkl,ockl->bohw', sampled_features, self.weight)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output

    def visualize_sampling_pattern(self, center_y=112, center_x=112):
        """
        Visualize the sampling pattern for a specific position
        
        Args:
            center_y: Y coordinate to visualize
            center_x: X coordinate to visualize
            
        Returns:
            sampling_positions: [kernel_size, kernel_size, 2] showing where each kernel position samples from
        """
        if center_y >= self.input_height or center_x >= self.input_width:
            raise ValueError(f"Center position ({center_y}, {center_x}) is outside input bounds")
            
        return self.foveated_kernels[center_y, center_x].clone()


class ConcreteFoveatedSamplingConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=11, input_height=256, input_width=256, center_size=3):
        """
        Concrete Foveated Sampling Convolution Layer
        Pre-computes foveated kernels for every position in the feature map
        
        Args:
            in_channels: Input channels (3 from your 1x1 conv)
            out_channels: Output feature channels
            kernel_size: Size of the foveated kernel (should be odd, e.g., 11x11)
            input_height: Height of input feature map
            input_width: Width of input feature map
            center_size: Size of the center high-resolution region (e.g., 1 for 1x1, 3 for 3x3, 5 for 5x5)
        """
        super(ConcreteFoveatedSamplingConv2d, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert center_size % 2 == 1, "Center size must be odd"
        assert center_size <= kernel_size, "Center size must be <= kernel_size"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_height = input_height
        self.input_width = input_width
        self.center = kernel_size // 2
        self.center_radius = center_size // 2  # Radius of center region
        
        # Define the sampling scales for each ring
        self.scales = {
            'center': 1,      # Direct sampling (red region)
            'ring1': 3,       # 9x area -> ~3x scale (orange region)
            'ring2': 5,       # 25x area -> ~5x scale (purple region)  
            'ring3': 7        # 49x area -> ~7x scale (light blue region)
        }
        
        # Learnable convolution weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Pre-compute all foveated sampling positions for each (y,x) in feature map
        self.register_buffer('foveated_kernels', self._create_all_foveated_kernels())
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_ring_type(self, i, j):
        """Determine which sampling ring a kernel position belongs to"""
        center = self.center
        dist = max(abs(i - center), abs(j - center))
        
        if dist <= self.center_radius:
            return 'center'  # Center region with configurable size
        elif dist <= self.center_radius + 1:
            return 'ring1' 
        elif dist <= self.center_radius + 2:
            return 'ring2'
        else:
            return 'ring3'
    
    def _create_all_foveated_kernels(self):
        """
        Pre-compute foveated sampling positions for every (y,x) in the feature map
        Returns tensor of shape [H, W, kernel_size, kernel_size, 2] containing (y,x) sampling coordinates
        """
        H, W = self.input_height, self.input_width
        kernels = torch.zeros(H, W, self.kernel_size, self.kernel_size, 2)
        
        for y in range(H):
            for x in range(W):
                for ki in range(self.kernel_size):
                    for kj in range(self.kernel_size):
                        # Determine which ring this kernel position belongs to
                        ring_type = self._get_ring_type(ki, kj)
                        scale = self.scales[ring_type]
                        
                        # Calculate the sampling position
                        # Offset from kernel center, scaled by ring
                        offset_y = (ki - self.center) * scale
                        offset_x = (kj - self.center) * scale
                        
                        # Final sampling position (clamped to image boundaries)
                        sample_y = max(0, min(H-1, y + offset_y))
                        sample_x = max(0, min(W-1, x + offset_x))
                        
                        kernels[y, x, ki, kj, 0] = sample_y
                        kernels[y, x, ki, kj, 1] = sample_x
        
        return kernels
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _sample_at_positions(self, x, sample_coords):
        """
        Sample from input feature map at specified coordinates using bilinear interpolation
        
        Args:
            x: Input tensor [B, C, H, W]
            sample_coords: Sampling coordinates [H, W, kernel_size, kernel_size, 2]
        
        Returns:
            sampled: [B, C, H, W, kernel_size, kernel_size]
        """
        B, C, H, W = x.shape
        
        # Flatten coordinates for grid_sample
        coords_flat = sample_coords.view(-1, 2)  # [H*W*K*K, 2]
        
        # Normalize coordinates to [-1, 1] for grid_sample
        coords_normalized = torch.zeros_like(coords_flat)
        coords_normalized[:, 0] = 2.0 * coords_flat[:, 1] / (W - 1) - 1.0  # x coordinates (note: x comes first in grid_sample)
        coords_normalized[:, 1] = 2.0 * coords_flat[:, 0] / (H - 1) - 1.0  # y coordinates
        
        # Reshape for grid_sample: [B, H*W*K*K, 1, 2]
        grid = coords_normalized.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
        
        # Sample features
        sampled_flat = F.grid_sample(
            x, 
            grid, 
            mode='bilinear', #bilinear or bicubic 
            padding_mode='border', 
            align_corners=True
        )  # [B, C, H*W*K*K, 1]
        
        # Reshape back to [B, C, H, W, K, K]
        sampled = sampled_flat.squeeze(-1).view(B, C, H, W, self.kernel_size, self.kernel_size)
        
        return sampled
    
    def forward(self, x):
        """
        Forward pass of concrete foveated sampling convolution
        
        Args:
            x: Input feature map [B, 3, H, W]
        
        Returns:
            output: [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        
        # Ensure input dimensions match what we pre-computed for
        assert H == self.input_height and W == self.input_width, \
            f"Input size ({H}, {W}) doesn't match expected ({self.input_height}, {self.input_width})"
        
        # Sample features at all pre-computed foveated positions
        # sampled_features: [B, C, H, W, kernel_size, kernel_size]
        sampled_features = self._sample_at_positions(x, self.foveated_kernels)
        
        # Apply convolution: multiply sampled features with learned weights
        # sampled_features: [B, C, H, W, K, K]
        # weight: [out_channels, in_channels, K, K]
        # Use einsum for efficient batched convolution
        output = torch.einsum('bchwkl,ockl->bohw', sampled_features, self.weight)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output

    def visualize_sampling_pattern(self, center_y=112, center_x=112):
        """
        Visualize the sampling pattern for a specific position
        
        Args:
            center_y: Y coordinate to visualize
            center_x: X coordinate to visualize
            
        Returns:
            sampling_positions: [kernel_size, kernel_size, 2] showing where each kernel position samples from
        """
        if center_y >= self.input_height or center_x >= self.input_width:
            raise ValueError(f"Center position ({center_y}, {center_x}) is outside input bounds")
            
        return self.foveated_kernels[center_y, center_x].clone()