"""
PackVisionNet - Custom Conditional Packaging Generative Neural Network
=====================================================================
A conditional deep neural network for sketch-to-photoreal packaging rendering:
- Multi-scale U-Net encoder-decoder architecture with skip connections.
- Feature-wise Linear Modulation (FiLM) conditioning on substrate materials and print finishes.
- Spatial self-attention in the bottleneck for long-range packaging perspective geometry.
- Multi-term loss: L1 pixel loss + Sobel edge consistency loss + Structural SSIM loss.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.
    
    Applies affine transformation conditioned on semantic packaging embeddings:
    y = gamma(cond) * x + beta(cond)
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.Linear(cond_dim, num_channels * 2),
            nn.SiLU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim) -> (B, num_channels * 2)
        params = self.modulation(cond)
        gamma, beta = params.chunk(2, dim=-1)
        # Reshape for broadcasting: (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class ResnetBlock(nn.Module):
    """Residual Convolutional Block with FiLM conditioning."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.film1 = FiLMBlock(cond_dim, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.film2 = FiLMBlock(cond_dim, out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film1(h, cond)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film2(h, cond)
        h = self.act2(h + res)
        return h


class SpatialSelfAttention(nn.Module):
    """Self-Attention block for capturing long-range perspective geometry."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, num_heads, H*W, head_dim)
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Downsampling with strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling with bilinear interpolation + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class PackVisionNet(nn.Module):
    """
    PackVisionNet: Conditional Packaging Image-to-Image Neural Architecture.
    
    Transforms 1-channel sketch line-art into 3-channel studio photoreal renders,
    conditioned on packaging category, substrate material, and print finish embeddings.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_channels: int = 32,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_categories: int = 16,
        num_materials: int = 16,
        num_finishes: int = 16,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # Semantic Conditioning Embeddings
        self.category_embed = nn.Embedding(num_categories, embedding_dim)
        self.material_embed = nn.Embedding(num_materials, embedding_dim)
        self.finish_embed = nn.Embedding(num_finishes, embedding_dim)
        
        # Total conditioning vector dimension: 3 * embedding_dim
        cond_dim = embedding_dim * 3
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Initial Feature Projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder stages
        self.downs = nn.ModuleList()
        current_channels = base_channels
        self.skip_channels = [current_channels]

        for i, mult in enumerate(channel_multipliers):
            target_channels = base_channels * mult
            block1 = ResnetBlock(current_channels, target_channels, cond_dim)
            block2 = ResnetBlock(target_channels, target_channels, cond_dim)
            self.downs.append(nn.ModuleList([block1, block2]))
            self.skip_channels.append(target_channels)
            current_channels = target_channels

            # Downsample between stages except the last
            if i < len(channel_multipliers) - 1:
                self.downs.append(Downsample(current_channels))

        # Bottleneck (Mid-Block with Spatial Attention)
        self.mid_block1 = ResnetBlock(current_channels, current_channels, cond_dim)
        self.mid_attn = SpatialSelfAttention(current_channels)
        self.mid_block2 = ResnetBlock(current_channels, current_channels, cond_dim)

        # Decoder stages (reverse)
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_multipliers)):
            target_channels = base_channels * mult
            # Skip connection from encoder has target_channels
            self.ups.append(ResnetBlock(current_channels + target_channels, target_channels, cond_dim))
            self.ups.append(ResnetBlock(target_channels, target_channels, cond_dim))
            current_channels = target_channels

            if i < len(channel_multipliers) - 1:
                self.ups.append(Upsample(current_channels))

        # Output Head
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def get_conditioning(
        self,
        category_idx: torch.Tensor,
        material_idx: torch.Tensor,
        finish_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds and projects categorical packaging selectors into a continuous conditioning vector."""
        cat_e = self.category_embed(category_idx)
        mat_e = self.material_embed(material_idx)
        fin_e = self.finish_embed(finish_idx)
        cond_raw = torch.cat([cat_e, mat_e, fin_e], dim=-1)
        return self.cond_mlp(cond_raw)

    def forward(
        self,
        sketch: torch.Tensor,
        category_idx: torch.Tensor,
        material_idx: torch.Tensor,
        finish_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of PackVisionNet.
        
        Args:
            sketch: (B, 1, H, W) normalized sketch lines in [0, 1].
            category_idx: (B,) integer tensor of packaging categories.
            material_idx: (B,) integer tensor of substrate materials.
            finish_idx: (B,) integer tensor of print finishes.
            
        Returns:
            (B, 3, H, W) rendered photoreal packaging image in [0, 1].
        """
        # 1. Conditioning vector
        cond = self.get_conditioning(category_idx, material_idx, finish_idx)

        # 2. Initial convolution
        x = self.init_conv(sketch)
        skips = [x]

        # 3. Encoder
        for module in self.downs:
            if isinstance(module, nn.ModuleList):
                x = module[0](x, cond)
                x = module[1](x, cond)
                skips.append(x)
            else:
                x = module(x)

        # 4. Bottleneck with Self-Attention
        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        # 5. Decoder with Skip Connections
        up_idx = 0
        while up_idx < len(self.ups):
            module = self.ups[up_idx]
            if isinstance(module, ResnetBlock):
                skip = skips.pop()
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = module(x, cond)
                up_idx += 1
                # Second block
                x = self.ups[up_idx](x, cond)
                up_idx += 1
            else:
                # Upsample
                x = module(x)
                up_idx += 1

        # 6. Final projection + Sigmoid [0, 1]
        x = self.out_norm(x)
        x = F.silu(x)
        x = torch.sigmoid(self.out_conv(x))
        return x


# =============================================================================
# PACKAGING COMPOSITE LOSS FUNCTION
# =============================================================================

class PackagingLoss(nn.Module):
    """
    Multi-objective loss function for photorealistic packaging synthesis:
    L_total = L1_loss + lambda_edge * SobelEdgeLoss + lambda_ssim * SSIMLoss
    """

    def __init__(self, lambda_edge: float = 0.5, lambda_ssim: float = 0.3):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_ssim = lambda_ssim

        # Sobel kernel for edge gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edge_gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient magnitude from RGB image."""
        # Convert RGB to grayscale
        gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Differentiable SSIM approximation."""
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
        sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2) + 1e-6
        )
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Pixel-level L1 reconstruction loss
        l1_loss = F.l1_loss(pred, target)

        # 2. Structural Sobel edge loss (preserves folds, creases, dielines)
        edge_pred = self._edge_gradient(pred)
        edge_target = self._edge_gradient(target)
        edge_loss = F.l1_loss(edge_pred, edge_target)

        # 3. Structural SSIM loss
        ssim_val = self._ssim(pred, target)
        ssim_loss = 1.0 - ssim_val

        # Composite total loss
        total_loss = l1_loss + self.lambda_edge * edge_loss + self.lambda_ssim * ssim_loss

        return {
            "loss": total_loss,
            "l1": l1_loss,
            "edge": edge_loss,
            "ssim": ssim_val,
        }
