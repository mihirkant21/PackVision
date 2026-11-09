"""
PackVisionNet - Custom Conditional Packaging Generative Neural Network (v2.0)
=============================================================================
A state-of-the-art conditional deep neural network for sketch-to-photoreal
packaging rendering:
- Multi-scale residual U-Net encoder-decoder architecture with gated skip connections.
- Dual conditioning:
    1. Categorical FiLM modulation (Category, Substrate Material, Print Finish embeddings).
    2. Structural Edge Attention (SEA) injecting multi-scale sketch line-art guidance
       directly into encoder stages, bottleneck, and decoder stages.
- Spatial self-attention in the bottleneck for long-range packaging 3D perspective geometry.
- Progressive residual refinement head for razor-sharp edge sharpening and specular definition.
- Multi-Scale PatchGAN Feature Discriminator for perceptual texture evaluation.
- Multi-term composite objective:
    L_total = lambda_l1 * L1 + lambda_ssim * (1 - SSIM) + lambda_edge * SobelEdge + lambda_color * ColorConsistency
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. CONDITIONING & MODULATION BLOCKS
# =============================================================================

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.
    
    Applies affine transformation conditioned on semantic packaging embeddings:
    y = (1.0 + gamma(cond)) * x + beta(cond)
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.Linear(cond_dim, num_channels * 2),
            nn.SiLU(),
            nn.Linear(num_channels * 2, num_channels * 2),
        )
        # Initialize last layer with zeros for identity modulation at init
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.modulation(cond)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class StructuralEdgeAttention(nn.Module):
    """Structural Edge Attention (SEA) block for packaging line-art guidance.
    
    Prevents faint or flat features when processing complex 3D sketches by
    dynamically routing edge boundary features into convolutional representations.
    Computes spatial gating and channel excitation conditioned on sketch dielines.
    """

    def __init__(self, in_channels: int, edge_channels: int):
        super().__init__()
        inter_channels = max(16, in_channels // 4)
        self.conv_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.conv_e = nn.Conv2d(edge_channels, inter_channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(edge_channels, in_channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor([0.2]))

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        if edge.shape[2:] != x.shape[2:]:
            edge = F.interpolate(edge, size=x.shape[2:], mode="bilinear", align_corners=False)
        g = self.gate(self.conv_x(x) + self.conv_e(edge))
        return x + self.scale * (x * g + self.proj(edge))


class AttentionGate(nn.Module):
    """Spatial Attention Gate for U-Net skip connections.
    
    Filters out background noise and reinforces structural packaging creases
    and panel silhouettes in skip features using gating signals from the decoder.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, F_int), F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, F_int), F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.SiLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        psi = self.act(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


# =============================================================================
# 2. CONVOLUTIONAL & ATTENTION BUILDING BLOCKS
# =============================================================================

class ResnetBlock(nn.Module):
    """Residual Convolutional Block with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        edge_channels: int = 0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.film1 = FiLMBlock(cond_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.film2 = FiLMBlock(cond_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

        if edge_channels > 0:
            self.edge_attn = StructuralEdgeAttention(out_channels, edge_channels)
        else:
            self.edge_attn = None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        edge: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = F.silu(self.film1(self.norm1(self.conv1(x)), cond))
        h = self.film2(self.norm2(self.conv2(h)), cond)
        out = F.silu(h + self.shortcut(x))

        if self.edge_attn is not None and edge is not None:
            out = self.edge_attn(out, edge)
        return out


class SpatialSelfAttention(nn.Module):
    """Optimized Self-Attention block for capturing long-range 3D perspective geometry.
    
    Uses memory-efficient scaled dot-product attention compatible with arbitrary resolutions.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, num_heads, H*W, head_dim)
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(q, k, v)
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


class SketchEdgePyramid(nn.Module):
    """Multi-scale structural edge feature extractor for line-art guidance."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU(),
        )

    def forward(self, sketch: torch.Tensor) -> List[torch.Tensor]:
        e0 = self.stage0(sketch)
        e1 = self.stage1(e0)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        return [e0, e1, e2, e3]


# =============================================================================
# 3. GENERATOR ARCHITECTURE: PACKVISIONNET (v2.0)
# =============================================================================

class PackVisionNet(nn.Module):
    """PackVisionNet-v2: Advanced Conditional Packaging Image-to-Image Neural Architecture.
    
    Transforms 1-channel sketch line-art into 3-channel studio photoreal renders,
    conditioned on packaging category, substrate material, and print finish embeddings.
    Features:
    - Multi-scale residual U-Net encoder-decoder with Attention-Gated skip connections.
    - Dual conditioning: Categorical FiLM modulation + Structural Edge Attention (SEA).
    - Bottleneck spatial self-attention for perspective dieline geometry.
    - Progressive residual refinement head for 512x512 edge sharpness and specular definition.
    - Native support for 256x256, 512x512, and variable aspect ratios.
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
        edge_attention: bool = True,
        use_attention_gate: bool = True,
        progressive_refinement: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.embedding_dim = embedding_dim
        self.edge_attention = edge_attention
        self.use_attention_gate = use_attention_gate
        self.progressive_refinement = progressive_refinement

        # 1. Semantic Conditioning Embeddings
        self.category_embed = nn.Embedding(num_categories, embedding_dim)
        self.material_embed = nn.Embedding(num_materials, embedding_dim)
        self.finish_embed = nn.Embedding(num_finishes, embedding_dim)

        cond_dim = embedding_dim * 3
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # 2. Structural Edge Guidance Pyramid
        self.edge_pyramid = SketchEdgePyramid(in_channels, base_channels) if edge_attention else None

        # 3. Initial Feature Projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # 4. Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.encoder_edge_attns = nn.ModuleList() if edge_attention else None
        current_channels = base_channels
        self.stage_channels = [current_channels]

        for i, mult in enumerate(channel_multipliers):
            target_channels = base_channels * mult
            b1 = ResnetBlock(current_channels, target_channels, cond_dim)
            b2 = ResnetBlock(target_channels, target_channels, cond_dim)
            down = Downsample(target_channels) if i < len(channel_multipliers) - 1 else nn.Identity()
            self.encoder_stages.append(nn.ModuleDict({
                "block1": b1,
                "block2": b2,
                "down": down,
            }))
            if edge_attention:
                self.encoder_edge_attns.append(
                    StructuralEdgeAttention(target_channels, target_channels)
                )
            self.stage_channels.append(target_channels)
            current_channels = target_channels

        # 5. Bottleneck (Mid-Block with Spatial Attention & Edge Attention)
        self.mid_block1 = ResnetBlock(current_channels, current_channels, cond_dim)
        self.mid_edge = StructuralEdgeAttention(current_channels, current_channels) if edge_attention else None
        self.mid_attn = SpatialSelfAttention(current_channels)
        self.mid_block2 = ResnetBlock(current_channels, current_channels, cond_dim)

        # 6. Decoder stages with Attention Gates & Edge Attention
        self.decoder_stages = nn.ModuleList()
        self.attn_gates = nn.ModuleList() if use_attention_gate else None
        self.decoder_edge_attns = nn.ModuleList() if edge_attention else None

        rev_multipliers = list(reversed(channel_multipliers))
        for i, mult in enumerate(rev_multipliers):
            target_channels = base_channels * mult
            skip_ch = target_channels
            if use_attention_gate:
                self.attn_gates.append(
                    AttentionGate(F_g=current_channels, F_l=skip_ch, F_int=max(16, skip_ch // 2))
                )
            b1 = ResnetBlock(current_channels + skip_ch, target_channels, cond_dim)
            b2 = ResnetBlock(target_channels, target_channels, cond_dim)
            up = Upsample(target_channels) if i < len(channel_multipliers) - 1 else nn.Identity()
            self.decoder_stages.append(nn.ModuleDict({
                "block1": b1,
                "block2": b2,
                "up": up,
            }))
            if edge_attention:
                self.decoder_edge_attns.append(
                    StructuralEdgeAttention(target_channels, target_channels)
                )
            current_channels = target_channels

        # 7. Output Head
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        # 8. Progressive Refinement Head (Residual Edge Sharpening)
        if progressive_refinement:
            self.refinement_head = nn.Sequential(
                nn.Conv2d(base_channels + in_channels, base_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.refinement_head[-1].weight)
            nn.init.zeros_(self.refinement_head[-1].bias)
        else:
            self.refinement_head = None

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
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass of PackVisionNet-v2.
        
        Args:
            sketch: (B, 1, H, W) normalized sketch lines in [0, 1].
            category_idx: (B,) integer tensor of packaging categories.
            material_idx: (B,) integer tensor of substrate materials.
            finish_idx: (B,) integer tensor of print finishes.
            return_aux: whether to return auxiliary intermediate maps for training.
            
        Returns:
            (B, 3, H, W) rendered photoreal packaging image in [0, 1].
        """
        # 1. Continuous conditioning vector
        cond = self.get_conditioning(category_idx, material_idx, finish_idx)

        # 2. Multi-scale edge guidance pyramid
        edge_feats = self.edge_pyramid(sketch) if self.edge_attention else [None] * 4

        # 3. Initial convolution
        x = self.init_conv(sketch)
        skips: List[torch.Tensor] = []

        # 4. Encoder passes
        for i, stage in enumerate(self.encoder_stages):
            x = stage["block1"](x, cond)
            x = stage["block2"](x, cond)
            if self.edge_attention:
                x = self.encoder_edge_attns[i](x, edge_feats[min(i, len(edge_feats) - 1)])
            skips.append(x)
            x = stage["down"](x)

        # 5. Bottleneck with Self-Attention & Edge Guidance
        x = self.mid_block1(x, cond)
        if self.edge_attention:
            x = self.mid_edge(x, edge_feats[-1])
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        # 6. Decoder passes with Gated Skip Connections & Edge Attention
        for i, stage in enumerate(self.decoder_stages):
            skip = skips.pop()
            if self.use_attention_gate:
                skip = self.attn_gates[i](x, skip)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = stage["block1"](x, cond)
            x = stage["block2"](x, cond)
            if self.edge_attention:
                x = self.decoder_edge_attns[i](x, edge_feats[len(self.decoder_stages) - 1 - i])
            x = stage["up"](x)

        # 7. Base Output Projection
        feat = F.silu(self.out_norm(x))
        base_out = self.out_conv(feat)

        # 8. Progressive Residual Edge Sharpening Head
        if self.refinement_head is not None:
            res_input = torch.cat([feat, sketch], dim=1)
            delta = self.refinement_head(res_input)
            out = torch.sigmoid(base_out + delta)
        else:
            out = torch.sigmoid(base_out)

        if return_aux:
            return out, {"base_out": torch.sigmoid(base_out), "feat": feat}
        return out


# =============================================================================
# 4. MULTI-SCALE PATCHGAN FEATURE DISCRIMINATOR
# =============================================================================

class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator for photorealistic packaging synthesis."""

    def __init__(self, in_channels: int = 4, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale PatchGAN Feature Discriminator operating at 1x and 0.5x scales."""

    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 3,
        num_discriminators: int = 2,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(in_channels=in_channels, ndf=ndf, n_layers=n_layers)
            for _ in range(num_discriminators)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        input_down = x
        for disc in self.discriminators:
            outputs.append(disc(input_down))
            input_down = self.downsample(input_down)
        return outputs


# =============================================================================
# 5. MULTI-TERM PERCEPTUAL & STRUCTURAL LOSS FUNCTION
# =============================================================================

class MultiScaleSobelLoss(nn.Module):
    """Multi-scale spatial edge gradient loss using horizontal and vertical Sobel kernels."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_magnitude(self, gray: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gray_p = 0.2989 * pred[:, 0:1] + 0.5870 * pred[:, 1:2] + 0.1140 * pred[:, 2:3]
        gray_t = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

        # 1x scale (fine details, folds, die-lines)
        loss_1x = F.l1_loss(self._gradient_magnitude(gray_p), self._gradient_magnitude(gray_t))

        # 0.5x scale (macro packaging contours and bevels)
        gray_p_half = F.avg_pool2d(gray_p, 2)
        gray_t_half = F.avg_pool2d(gray_t, 2)
        loss_half = F.l1_loss(self._gradient_magnitude(gray_p_half), self._gradient_magnitude(gray_t_half))

        return 0.7 * loss_1x + 0.3 * loss_half


class ColorConsistencyLoss(nn.Module):
    """Chroma and color distribution loss for substrate fidelity."""

    def __init__(self, blur_kernel_size: int = 8):
        super().__init__()
        self.kernel_size = blur_kernel_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Low-frequency spatial color blur matching
        blur_p = F.avg_pool2d(pred, self.kernel_size, stride=4, padding=self.kernel_size // 4)
        blur_t = F.avg_pool2d(target, self.kernel_size, stride=4, padding=self.kernel_size // 4)
        l_blur = F.l1_loss(blur_p, blur_t)

        # 2. Global channel mean & standard deviation matching
        mu_p = pred.mean(dim=[-2, -1])
        mu_t = target.mean(dim=[-2, -1])
        std_p = pred.std(dim=[-2, -1])
        std_t = target.std(dim=[-2, -1])
        l_stat = F.l1_loss(mu_p, mu_t) + 0.5 * F.l1_loss(std_p, std_t)

        return 0.7 * l_blur + 0.3 * l_stat


class PackagingLoss(nn.Module):
    """Composite loss function for photorealistic packaging synthesis:
    
    L_total = lambda_l1 * L_L1 + lambda_ssim * (1 - SSIM) + lambda_edge * L_Sobel + lambda_color * L_color
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_edge: float = 0.4,
        lambda_color: float = 0.3,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_color = lambda_color

        self.sobel_loss = MultiScaleSobelLoss()
        self.color_loss = ColorConsistencyLoss()

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

        # 2. Structural SSIM loss
        ssim_val = self._ssim(pred, target)
        ssim_loss = 1.0 - ssim_val

        # 3. Structural Sobel edge gradient loss
        edge_loss = self.sobel_loss(pred, target)

        # 4. Color / chroma substrate consistency loss
        color_loss = self.color_loss(pred, target)

        # Total Composite Objective
        total_loss = (
            self.lambda_l1 * l1_loss
            + self.lambda_ssim * ssim_loss
            + self.lambda_edge * edge_loss
            + self.lambda_color * color_loss
        )

        return {
            "loss": total_loss,
            "l1": l1_loss,
            "ssim": ssim_val,
            "ssim_loss": ssim_loss,
            "edge": edge_loss,
            "color": color_loss,
        }
