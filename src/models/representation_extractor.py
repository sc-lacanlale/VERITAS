"""
Multi-representation feature extraction for VERITAS.

Extracts RGB, noise residual, and frequency-domain representations
for forensic image analysis.

Requirements: 6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import dct
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class RGBExtractor(nn.Module):
    """
    Extract RGB representation.
    
    This is simply the normalized RGB image (already handled in preprocessing).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Get normalization parameters
        self.mean = torch.tensor(
            config.get('representations.imagenet_mean', [0.485, 0.456, 0.406])
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            config.get('representations.imagenet_std', [0.229, 0.224, 0.225])
        ).view(1, 3, 1, 1)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract RGB representation.
        
        Args:
            image: Input tensor [B, 3, H, W] (already normalized)
            
        Returns:
            RGB representation [B, 3, H, W]
        """
        # Image is already normalized during preprocessing
        # Just return as-is
        return image
    
    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Denormalize image for visualization.
        
        Args:
            image: Normalized image [B, 3, H, W]
            
        Returns:
            Denormalized image [B, 3, H, W]
        """
        self.mean = self.mean.to(image.device)
        self.std = self.std.to(image.device)
        return image * self.std + self.mean


class NoiseResidualExtractor(nn.Module):
    """
    Extract noise residual representation using SRM (Spatial Rich Model) filters.
    
    SRM filters are designed to suppress image content and highlight
    subtle manipulation artifacts like resampling, JPEG compression, etc.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create SRM filters
        self.num_filters = config.get('representations.srm_filter_count', 3)
        self.register_buffer('srm_filters', self._create_srm_filters())
        
        # Normalization type
        self.norm_type = config.get('representations.noise_normalization', 'tanh')
    
    def _create_srm_filters(self) -> torch.Tensor:
        """
        Create SRM filter bank.
        
        Uses 3 classic SRM filters from the 30-filter SRM set.
        These are 3x3 kernels designed to detect high-frequency artifacts.
        
        Returns:
            Tensor of SRM filters [num_filters, 1, 3, 3]
        """
        # SRM Filter 1: Basic high-pass filter
        filter1 = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=np.float32) / 4.0
        
        # SRM Filter 2: Edge detector (horizontal)
        filter2 = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32)
        # Pad to 3x3 for consistency (or use the center 3x3)
        filter2 = filter2[1:4, 1:4] / 12.0
        
        # SRM Filter 3: Edge detector (diagonal)
        filter3 = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=np.float32) / 4.0
        filter3 = np.rot90(filter3)  # Rotate 90 degrees
        
        # Stack filters
        filters = np.stack([filter1, filter2, filter3])  # [3, 3, 3]
        filters = filters[:, np.newaxis, :, :]  # [3, 1, 3, 3]
        
        return torch.from_numpy(filters).float()
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract noise residual representation.
        
        Args:
            image: Input tensor [B, 3, H, W] (normalized)
            
        Returns:
            Noise residual representation [B, 3, H, W]
        """
        B, C, H, W = image.shape
        
        # Denormalize image first (SRM works on [0, 1] range)
        # Convert from ImageNet normalized to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        image_01 = image * std + mean  # Denormalize to [0, 1] range
        
        # Convert to grayscale for SRM filter application
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * image_01[:, 0:1] + 0.587 * image_01[:, 1:2] + 0.114 * image_01[:, 2:3]
        # gray shape: [B, 1, H, W]
        
        # Apply SRM filters
        noise_residuals = []
        for i in range(self.num_filters):
            # Get filter
            srm_filter = self.srm_filters[i:i+1]  # [1, 1, 3, 3]
            
            # Convolve
            residual = F.conv2d(
                gray,
                srm_filter,
                padding=1  # Same padding to preserve size
            )
            
            noise_residuals.append(residual)
        
        # Stack residuals to create 3-channel output
        noise = torch.cat(noise_residuals, dim=1)  # [B, 3, H, W]
        
        # Normalize to [-1, 1] range using tanh
        if self.norm_type == 'tanh':
            noise = torch.tanh(noise)
        elif self.norm_type == 'sigmoid':
            noise = torch.sigmoid(noise) * 2 - 1  # Map [0,1] to [-1,1]
        else:
            # Clip to [-1, 1]
            noise = torch.clamp(noise, -1, 1)
        
        return noise


class FrequencyExtractor(nn.Module):
    """
    Extract frequency-domain representation using DCT.
    
    JPEG compression uses 8×8 DCT blocks, so manipulation often
    leaves artifacts in the DCT domain.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # DCT parameters
        self.block_size = config.get('representations.dct_block_size', 8)
        self.frequency_bands = config.get('representations.frequency_bands', ['low', 'mid', 'high'])
    
    def _extract_dct_blocks(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply 8×8 block DCT to image.
        
        Args:
            image: Input tensor [B, C, H, W]
            
        Returns:
            DCT coefficients [B, C, H, W]
        """
        B, C, H, W = image.shape
        
        # Convert to numpy for scipy DCT
        image_np = image.cpu().numpy()
        
        dct_result = np.zeros_like(image_np)
        
        # Process each batch and channel
        for b in range(B):
            for c in range(C):
                img = image_np[b, c]
                
                # Apply DCT to 8×8 blocks
                for i in range(0, H, self.block_size):
                    for j in range(0, W, self.block_size):
                        # Extract block
                        block = img[i:i+self.block_size, j:j+self.block_size]
                        
                        # Skip if block is not full size (edge cases)
                        if block.shape != (self.block_size, self.block_size):
                            dct_result[b, c, i:i+block.shape[0], j:j+block.shape[1]] = block
                            continue
                        
                        # Apply 2D DCT
                        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                        
                        # Store result
                        dct_result[b, c, i:i+self.block_size, j:j+self.block_size] = dct_block
        
        return torch.from_numpy(dct_result).to(image.device)
    
    def _extract_frequency_bands(self, dct_coeffs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract low, mid, and high frequency bands from DCT coefficients.
        
        Args:
            dct_coeffs: DCT coefficients [B, C, H, W]
            
        Returns:
            Dictionary with 'low', 'mid', 'high' frequency bands
        """
        B, C, H, W = dct_coeffs.shape
        block_size = self.block_size
        
        # Create masks for each frequency band
        low_freq = torch.zeros_like(dct_coeffs)
        mid_freq = torch.zeros_like(dct_coeffs)
        high_freq = torch.zeros_like(dct_coeffs)
        
        # For each 8×8 block, classify DCT coefficients by frequency
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                block = dct_coeffs[:, :, i:i+block_size, j:j+block_size]
                
                if block.shape[2:] != (block_size, block_size):
                    continue
                
                # Low frequency: top-left 3×3 (DC + low-freq AC)
                low_freq[:, :, i:i+3, j:j+3] = block[:, :, :3, :3]
                
                # Mid frequency: next ring (3-5)
                mid_freq[:, :, i:i+5, j:j+5] = block[:, :, :5, :5]
                mid_freq[:, :, i:i+3, j:j+3] = 0  # Subtract low freq region
                
                # High frequency: remaining (5-8)
                high_freq[:, :, i:i+block_size, j:j+block_size] = block
                high_freq[:, :, i:i+5, j:j+5] = 0  # Subtract low+mid freq regions
        
        return {
            'low': low_freq,
            'mid': mid_freq,
            'high': high_freq
        }
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency-domain representation.
        
        Args:
            image: Input tensor [B, 3, H, W] (normalized)
            
        Returns:
            Frequency representation [B, 3, H, W]
        """
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        image_01 = image * std + mean  # [0, 1] range
        
        # Convert to grayscale for DCT
        gray = 0.299 * image_01[:, 0:1] + 0.587 * image_01[:, 1:2] + 0.114 * image_01[:, 2:3]
        
        # Apply DCT
        dct_coeffs = self._extract_dct_blocks(gray)
        
        # Extract frequency bands
        bands = self._extract_frequency_bands(dct_coeffs)
        
        # Stack frequency bands as 3-channel output
        # Each channel represents one frequency band
        freq_repr = torch.cat([
            bands['low'],
            bands['mid'],
            bands['high']
        ], dim=1)  # [B, 3, H, W]
        
        # Normalize to [0, 1] range
        freq_repr = torch.abs(freq_repr)  # Take absolute values
        freq_repr = freq_repr / (torch.max(freq_repr) + 1e-7)  # Normalize
        
        return freq_repr


class RepresentationExtractor(nn.Module):
    """
    Complete multi-representation feature extractor.
    
    Extracts RGB, noise residual, and frequency-domain representations
    for comprehensive forensic analysis.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Create extractors
        self.rgb_extractor = RGBExtractor(config)
        self.noise_extractor = NoiseResidualExtractor(config)
        self.frequency_extractor = FrequencyExtractor(config)
        
        logger.info("Initialized RepresentationExtractor with RGB, Noise, and Frequency extractors")
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all representations.
        
        Args:
            image: Input tensor [B, 3, H, W] (normalized)
            
        Returns:
            Dictionary with 'rgb', 'noise', 'frequency' representations
            Each representation is [B, 3, H, W]
        """
        rgb = self.rgb_extractor(image)
        noise = self.noise_extractor(image)
        frequency = self.frequency_extractor(image)
        
        return {
            'rgb': rgb,
            'noise': noise,
            'frequency': frequency
        }
    
    def get_representation_stats(self, representations: Dict[str, torch.Tensor]) -> Dict:
        """
        Get statistics for each representation (for debugging).
        
        Args:
            representations: Dictionary of representations
            
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        for name, tensor in representations.items():
            stats[name] = {
                'shape': tuple(tensor.shape),
                'min': tensor.min().item(),
                'max': tensor.max().item(),
                'mean': tensor.mean().item(),
                'std': tensor.std().item()
            }
        
        return stats
