import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FoveatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, aggregation='mean', padding_mode='reflect'):
        """
        Foveated Convolution Layer
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            aggregation: 'mean', 'max', 'min', or 'median'
            padding_mode: PyTorch padding mode for boundary handling
        """
        super(FoveatedConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation
        self.padding_mode = padding_mode
        
        # Total kernel size is 7x7, so we need padding of 3
        self.padding = 3
        
        # Define the number of values each position will aggregate
        # Inner 3x3: direct values (1 value each)
        # 5x5 ring: 5 values for edges, 10 for corners  
        # 7x7 ring: 15 values for top/bottom edges, 25 for left/right edges, 40 for corners
        
        # Initialize learnable convolution weights 
        # We'll have 49 total features per spatial position: 9 (inner) + 16 (5x5 ring) + 24 (7x7 ring)
        # This creates a 1x1 convolution on the 49 aggregated features
        self.conv = nn.Conv2d(in_channels * 49, out_channels, kernel_size=1, bias=True)
        
        # Pre-compute sampling positions for efficiency
        self._generate_sampling_positions()
        
    def _generate_sampling_positions(self):
        """Generate all sampling positions for the foveated kernel"""
        
        # Center position (3,3) in 7x7 kernel
        center = 3
        
        # Storage for sampling positions
        # Each element will be a list of (row_offset, col_offset) tuples
        self.sampling_positions = {}
        
        # Inner 3x3: Direct sampling (1 position each)
        for i in range(-1, 2):  # -1, 0, 1
            for j in range(-1, 2):  # -1, 0, 1
                key = f"inner_{i+1}_{j+1}"
                self.sampling_positions[key] = [(i, j)]
        
        # 5x5 ring positions
        ring_5x5_positions = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) == 2 or abs(j) == 2:  # Only outer ring
                    ring_5x5_positions.append((i, j))
        
        # Generate sampling patterns for 5x5 ring
        for pos_idx, (i, j) in enumerate(ring_5x5_positions):
            key = f"ring5_{pos_idx}"
            positions = []
            
            # Determine sampling direction based on position
            if i == -2 and abs(j) <= 1:  # Top edge
                positions = [(i + k, j) for k in range(5)]  # Sample downward
            elif i == 2 and abs(j) <= 1:   # Bottom edge  
                positions = [(i - k, j) for k in range(5)]  # Sample upward
            elif j == -2 and abs(i) <= 1:  # Left edge
                positions = [(i, j + k) for k in range(5)]  # Sample rightward
            elif j == 2 and abs(i) <= 1:   # Right edge
                positions = [(i, j - k) for k in range(5)]  # Sample leftward
            elif i == -2 and j == -2:      # Top-left corner
                positions = [(i + k, j) for k in range(5)] + [(i, j + k) for k in range(1, 5)]
            elif i == -2 and j == 2:       # Top-right corner
                positions = [(i + k, j) for k in range(5)] + [(i, j - k) for k in range(1, 5)]
            elif i == 2 and j == -2:       # Bottom-left corner
                positions = [(i - k, j) for k in range(5)] + [(i, j + k) for k in range(1, 5)]
            elif i == 2 and j == 2:        # Bottom-right corner
                positions = [(i - k, j) for k in range(5)] + [(i, j - k) for k in range(1, 5)]
                
            self.sampling_positions[key] = positions
            
        # 7x7 ring positions  
        ring_7x7_positions = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                if abs(i) == 3 or abs(j) == 3:  # Only outermost ring
                    ring_7x7_positions.append((i, j))
                    
        # Generate sampling patterns for 7x7 ring
        for pos_idx, (i, j) in enumerate(ring_7x7_positions):
            key = f"ring7_{pos_idx}"
            positions = []
            
            if i == -3 and abs(j) <= 2:    # Top edge
                positions = [(i + k, j) for k in range(15)]  # Sample downward  
            elif i == 3 and abs(j) <= 2:   # Bottom edge
                positions = [(i - k, j) for k in range(15)]  # Sample upward
            elif j == -3 and abs(i) <= 2:  # Left edge  
                positions = [(i, j + k) for k in range(25)]  # Sample rightward
            elif j == 3 and abs(i) <= 2:   # Right edge
                positions = [(i, j - k) for k in range(25)]  # Sample leftward
            elif i == -3 and j == -3:      # Top-left corner
                positions = [(i + k, j) for k in range(15)] + [(i, j + k) for k in range(1, 25)]
            elif i == -3 and j == 3:       # Top-right corner  
                positions = [(i + k, j) for k in range(15)] + [(i, j - k) for k in range(1, 25)]
            elif i == 3 and j == -3:       # Bottom-left corner
                positions = [(i - k, j) for k in range(15)] + [(i, j + k) for k in range(1, 25)]
            elif i == 3 and j == 3:        # Bottom-right corner
                positions = [(i - k, j) for k in range(15)] + [(i, j - k) for k in range(1, 25)]
                
            self.sampling_positions[key] = positions
    
    def _aggregate_samples(self, samples):
        """Aggregate sampled values according to specified method"""
        if self.aggregation == 'mean':
            return torch.mean(samples, dim=-1)
        elif self.aggregation == 'max':
            return torch.max(samples, dim=-1)[0]
        elif self.aggregation == 'min':
            return torch.min(samples, dim=-1)[0]
        elif self.aggregation == 'median':
            return torch.median(samples, dim=-1)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def forward(self, x):
        """
        Forward pass of foveated convolution
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        batch_size, in_channels, height, width = x.shape
        
        # Add padding to handle boundary cases
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 
                        mode=self.padding_mode)
        
        # Initialize output tensor
        foveated_features = torch.zeros(batch_size, in_channels, height, width, 49, 
                                      device=x.device, dtype=x.dtype)
        
        # Process each position in the output
        for h in range(height):
            for w in range(width):
                # Adjust for padding
                h_pad, w_pad = h + self.padding, w + self.padding
                feature_idx = 0
                
                # Inner 3x3: Direct sampling
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        sample_h = h_pad + i
                        sample_w = w_pad + j
                        foveated_features[:, :, h, w, feature_idx] = x_padded[:, :, sample_h, sample_w]
                        feature_idx += 1
                
                # 5x5 ring: Aggregated sampling
                ring_5x5_positions = []
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if abs(i) == 2 or abs(j) == 2:
                            ring_5x5_positions.append((i, j))
                
                for pos_idx, (i, j) in enumerate(ring_5x5_positions):
                    key = f"ring5_{pos_idx}"
                    sample_positions = self.sampling_positions[key]
                    
                    # Collect samples
                    samples = []
                    for di, dj in sample_positions:
                        sample_h = max(0, min(x_padded.shape[2] - 1, h_pad + di))
                        sample_w = max(0, min(x_padded.shape[3] - 1, w_pad + dj))
                        samples.append(x_padded[:, :, sample_h, sample_w])
                    
                    # Aggregate samples
                    if samples:
                        samples_tensor = torch.stack(samples, dim=-1)
                        aggregated = self._aggregate_samples(samples_tensor)
                        foveated_features[:, :, h, w, feature_idx] = aggregated
                    
                    feature_idx += 1
                
                # 7x7 ring: Aggregated sampling  
                ring_7x7_positions = []
                for i in range(-3, 4):
                    for j in range(-3, 4):
                        if abs(i) == 3 or abs(j) == 3:
                            ring_7x7_positions.append((i, j))
                
                for pos_idx, (i, j) in enumerate(ring_7x7_positions):
                    key = f"ring7_{pos_idx}"
                    sample_positions = self.sampling_positions[key]
                    
                    # Collect samples
                    samples = []
                    for di, dj in sample_positions:
                        sample_h = max(0, min(x_padded.shape[2] - 1, h_pad + di))
                        sample_w = max(0, min(x_padded.shape[3] - 1, w_pad + dj))
                        samples.append(x_padded[:, :, sample_h, sample_w])
                    
                    # Aggregate samples
                    if samples:
                        samples_tensor = torch.stack(samples, dim=-1)
                        aggregated = self._aggregate_samples(samples_tensor)
                        foveated_features[:, :, h, w, feature_idx] = aggregated
                    
                    feature_idx += 1
        
        # Reshape for convolution: (B, C, H, W, 49) -> (B, C*49, H, W)
        foveated_features = foveated_features.permute(0, 1, 4, 2, 3)  # (B, C, 49, H, W)
        foveated_features = foveated_features.contiguous().view(batch_size, in_channels * 49, height, width)
        
        # Apply 1x1 convolution to transform features
        output = self.conv(foveated_features)
        
        return output
    
    
class FastFoveatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, aggregation='mean', padding_mode='reflect'):
        """
        Fast Foveated Convolution Layer using vectorized operations
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            aggregation: 'mean', 'max', 'min', or 'median'
            padding_mode: PyTorch padding mode for boundary handling
        """
        super(FastFoveatedConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation
        self.padding_mode = padding_mode
        
        # Total kernel size is 7x7, so we need padding of 3
        self.padding = 3
        
        # Initialize learnable convolution weights 
        # We'll have 49 total features per spatial position: 9 (inner) + 16 (5x5 ring) + 24 (7x7 ring)
        # This creates a 1x1 convolution on the 49 aggregated features
        self.conv = nn.Conv2d(in_channels * 49, out_channels, kernel_size=1, bias=True)
        
        # Pre-compute all sampling coordinates for efficiency
        self._generate_sampling_coordinates()
        
    def _generate_sampling_coordinates(self):
        """Generate vectorized sampling coordinates for all 49 positions"""
        
        # Storage for sampling coordinates
        # Each element will contain the offsets for that position
        self.sampling_coords = []  # List of (num_samples, 2) tensors
        self.position_types = []   # Track which type each position is
        
        # Inner 3x3: Direct sampling (1 sample each)
        for i in range(-1, 2):
            for j in range(-1, 2):
                coords = torch.tensor([[i, j]], dtype=torch.long)
                self.sampling_coords.append(coords)
                self.position_types.append('inner')
        
        # 5x5 ring positions
        ring_5x5_positions = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if abs(i) == 2 or abs(j) == 2:  # Only outer ring
                    ring_5x5_positions.append((i, j))
        
        # Generate sampling patterns for 5x5 ring
        for i, j in ring_5x5_positions:
            coords = []
            
            # Determine sampling direction based on position
            if i == -2 and abs(j) <= 1:  # Top edge
                coords = [[i + k, j] for k in range(5)]  # Sample downward
            elif i == 2 and abs(j) <= 1:   # Bottom edge  
                coords = [[i - k, j] for k in range(5)]  # Sample upward
            elif j == -2 and abs(i) <= 1:  # Left edge
                coords = [[i, j + k] for k in range(5)]  # Sample rightward
            elif j == 2 and abs(i) <= 1:   # Right edge
                coords = [[i, j - k] for k in range(5)]  # Sample leftward
            elif i == -2 and j == -2:      # Top-left corner
                coords = [[i + k, j] for k in range(5)] + [[i, j + k] for k in range(1, 5)]
            elif i == -2 and j == 2:       # Top-right corner
                coords = [[i + k, j] for k in range(5)] + [[i, j - k] for k in range(1, 5)]
            elif i == 2 and j == -2:       # Bottom-left corner
                coords = [[i - k, j] for k in range(5)] + [[i, j + k] for k in range(1, 5)]
            elif i == 2 and j == 2:        # Bottom-right corner
                coords = [[i - k, j] for k in range(5)] + [[i, j - k] for k in range(1, 5)]
                
            coords_tensor = torch.tensor(coords, dtype=torch.long)
            self.sampling_coords.append(coords_tensor)
            self.position_types.append('ring5')
            
        # 7x7 ring positions  
        ring_7x7_positions = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                if abs(i) == 3 or abs(j) == 3:  # Only outermost ring
                    ring_7x7_positions.append((i, j))
                    
        # Generate sampling patterns for 7x7 ring
        for i, j in ring_7x7_positions:
            coords = []
            
            if i == -3 and abs(j) <= 2:    # Top edge
                coords = [[i + k, j] for k in range(15)]  # Sample downward  
            elif i == 3 and abs(j) <= 2:   # Bottom edge
                coords = [[i - k, j] for k in range(15)]  # Sample upward
            elif j == -3 and abs(i) <= 2:  # Left edge  
                coords = [[i, j + k] for k in range(25)]  # Sample rightward
            elif j == 3 and abs(i) <= 2:   # Right edge
                coords = [[i, j - k] for k in range(25)]  # Sample leftward
            elif i == -3 and j == -3:      # Top-left corner
                coords = [[i + k, j] for k in range(15)] + [[i, j + k] for k in range(1, 25)]
            elif i == -3 and j == 3:       # Top-right corner  
                coords = [[i + k, j] for k in range(15)] + [[i, j - k] for k in range(1, 25)]
            elif i == 3 and j == -3:       # Bottom-left corner
                coords = [[i - k, j] for k in range(15)] + [[i, j + k] for k in range(1, 25)]
            elif i == 3 and j == 3:        # Bottom-right corner
                coords = [[i - k, j] for k in range(15)] + [[i, j - k] for k in range(1, 25)]
                
            coords_tensor = torch.tensor(coords, dtype=torch.long)
            self.sampling_coords.append(coords_tensor)
            self.position_types.append('ring7')
    
    def _vectorized_sample_and_aggregate(self, x_padded, height, width):
        """Vectorized sampling and aggregation for all positions"""
        batch_size, in_channels = x_padded.shape[:2]
        device = x_padded.device
        
        # Pre-allocate output tensor
        features = torch.zeros(batch_size, in_channels, height, width, 49, 
                              device=device, dtype=x_padded.dtype)
        
        # Create coordinate grids for all spatial positions
        h_coords = torch.arange(height, device=device) + self.padding  # Adjust for padding
        w_coords = torch.arange(width, device=device) + self.padding
        h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
        
        # Process each of the 49 kernel positions
        for pos_idx, (coords, pos_type) in enumerate(zip(self.sampling_coords, self.position_types)):
            coords = coords.to(device)
            
            if pos_type == 'inner':
                # Direct sampling for inner 3x3
                offset_h, offset_w = coords[0]
                sample_h = h_grid + offset_h
                sample_w = w_grid + offset_w
                
                # Clamp coordinates to valid range
                sample_h = torch.clamp(sample_h, 0, x_padded.shape[2] - 1)
                sample_w = torch.clamp(sample_w, 0, x_padded.shape[3] - 1)
                
                # Advanced indexing to sample all positions at once
                features[:, :, :, :, pos_idx] = x_padded[:, :, sample_h, sample_w]
                
            else:
                # Aggregated sampling for outer rings
                samples_list = []
                
                for offset_h, offset_w in coords:
                    sample_h = h_grid + offset_h
                    sample_w = w_grid + offset_w
                    
                    # Clamp coordinates to valid range
                    sample_h = torch.clamp(sample_h, 0, x_padded.shape[2] - 1)
                    sample_w = torch.clamp(sample_w, 0, x_padded.shape[3] - 1)
                    
                    # Sample all positions for this offset
                    sampled = x_padded[:, :, sample_h, sample_w]
                    samples_list.append(sampled)
                
                # Stack samples and aggregate
                if samples_list:
                    samples = torch.stack(samples_list, dim=-1)  # (B, C, H, W, num_samples)
                    
                    # Apply aggregation
                    if self.aggregation == 'mean':
                        aggregated = torch.mean(samples, dim=-1)
                    elif self.aggregation == 'max':
                        aggregated = torch.max(samples, dim=-1)[0]
                    elif self.aggregation == 'min':
                        aggregated = torch.min(samples, dim=-1)[0]
                    elif self.aggregation == 'median':
                        aggregated = torch.median(samples, dim=-1)[0]
                    
                    features[:, :, :, :, pos_idx] = aggregated
        
        return features
    
    def forward(self, x):
        """
        Fast forward pass using vectorized operations
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        batch_size, in_channels, height, width = x.shape
        
        # Add padding to handle boundary cases
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), 
                        mode=self.padding_mode)
        
        # Vectorized sampling and aggregation
        foveated_features = self._vectorized_sample_and_aggregate(x_padded, height, width)
        
        # Reshape for convolution: (B, C, H, W, 49) -> (B, C*49, H, W)
        foveated_features = foveated_features.permute(0, 1, 4, 2, 3)  # (B, C, 49, H, W)
        foveated_features = foveated_features.contiguous().view(batch_size, in_channels * 49, height, width)
        
        # Apply 1x1 convolution to transform features
        output = self.conv(foveated_features)
        
        return output
    
class UltraFastFoveatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, aggregation='mean'):
        super(UltraFastFoveatedConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregation = aggregation
        
        # Create multiple conv layers for different sampling patterns
        # Inner 3x3: Standard convolution
        self.inner_conv = nn.Conv2d(in_channels, in_channels * 9, kernel_size=3, padding=1, groups=in_channels, bias=True)
        
        # Initialize inner conv to extract direct values
        # with torch.no_grad():
        #     # Identity mapping for 3x3 direct extraction
        #     for i in range(in_channels):
        #         self.inner_conv.weight[i*9:(i+1)*9, 0] = torch.eye(3).flatten().unsqueeze(0)
                
        with torch.no_grad():
            for c in range(self.in_channels):
                for k in range(9):
                    weight = torch.zeros(3, 3)
                    row = k // 3
                    col = k % 3
                    weight[row, col] = 1.0
                    # Assign the kernel to the correct output channel
                    self.inner_conv.weight[c * 9 + k, 0] = weight

        
        # Outer rings: Will use custom kernels for aggregation patterns
        # For simplicity, using approximation with dilated convolutions
        self.ring5_conv = nn.Conv2d(in_channels, in_channels * 16, kernel_size=5, padding=2, groups=in_channels, bias=True)
        self.ring7_conv = nn.Conv2d(in_channels, in_channels * 24, kernel_size=7, padding=3, groups=in_channels, bias=True)
        
        # Final combination
        self.final_conv = nn.Conv2d(in_channels * 49, out_channels, kernel_size=1, bias=True)
        
        self._init_custom_kernels()
    
    def _init_custom_kernels(self):
        """Initialize custom kernels for ring sampling patterns"""
        # This is a simplified version - you'd need to implement exact sampling patterns
        # For now, using averaging kernels as approximation
        
        with torch.no_grad():
            # Ring 5x5: averaging patterns
            for i in range(self.in_channels):
                for j in range(16):
                    # Create simple averaging kernels (approximation)
                    kernel = torch.zeros(5, 5)
                    # Fill based on position in ring...
                    self.ring5_conv.weight[i*16 + j, 0] = kernel
            
            # Ring 7x7: averaging patterns  
            for i in range(self.in_channels):
                for j in range(24):
                    # Create simple averaging kernels (approximation)
                    kernel = torch.zeros(7, 7)
                    # Fill based on position in ring...
                    self.ring7_conv.weight[i*24 + j, 0] = kernel
    
    def forward(self, x):
        # Extract features from each ring
        inner_features = self.inner_conv(x)    # (B, C*9, H, W)
        ring5_features = self.ring5_conv(x)    # (B, C*16, H, W)  
        ring7_features = self.ring7_conv(x)    # (B, C*24, H, W)
        
        # Concatenate all features
        all_features = torch.cat([inner_features, ring5_features, ring7_features], dim=1)
        
        # Final convolution
        output = self.final_conv(all_features)
        
        return output