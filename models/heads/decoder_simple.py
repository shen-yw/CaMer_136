import math

import torch
import torch.nn.functional as F
from torch import nn

from rvsd.models.temporal import TemporalFeatureFusion

from .dpt_head import ConvNormAct


class DecoderSimpleHead(nn.Module):
    """
    Legacy sparse decoder used by the earlier baseline/merge experiments.

    It keeps a lightweight dense skip path from pre-merge blocks, rasterizes the
    sparse tokens to a coarse grid late, then fuses both streams for final mask
    decoding.
    """

    def __init__(
        self,
        in_channels: int,
        decoder_dim: int = 192,
        output_dim: int = 128,
        use_batchnorm: bool = True,
        num_dense_sources: int = 2,
        sparse_coarse_stride: int = 2,
        sparse_temporal_layers: int = 1,
    ):
        super().__init__()
        if sparse_coarse_stride < 1:
            raise ValueError(f"sparse_coarse_stride must be >= 1, got {sparse_coarse_stride}")
        if num_dense_sources not in {1, 2}:
            raise ValueError(f"num_dense_sources must be 1 or 2, got {num_dense_sources}")
        self.num_dense_sources = num_dense_sources
        self.sparse_coarse_stride = sparse_coarse_stride
        self.detail_proj = nn.Conv2d(in_channels, decoder_dim, kernel_size=1)
        self.skip_proj = nn.Conv2d(in_channels, decoder_dim, kernel_size=1) if num_dense_sources == 2 else None
        self.sparse_proj = nn.Linear(in_channels, decoder_dim)
        self.sparse_fusion = TemporalFeatureFusion(
            in_channels=[decoder_dim],
            kernel_size=3,
            use_residual=True,
            use_gating=True,
            num_layers_per_scale=sparse_temporal_layers,
        )
        self.fuse = nn.Sequential(
            ConvNormAct(decoder_dim * (self.num_dense_sources + 1), decoder_dim, use_batchnorm=use_batchnorm),
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
        )
        self.refine = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            ConvNormAct(decoder_dim, output_dim, use_batchnorm=use_batchnorm),
        )
        self.output_upsample = nn.ConvTranspose2d(output_dim, output_dim, kernel_size=4, stride=4)

    def _rasterize_sparse_tokens(
        self,
        sparse_tokens: torch.Tensor,
        sparse_coords: torch.Tensor,
        sparse_sizes: torch.Tensor,
        patch_hw: tuple[int, int],
    ) -> torch.Tensor:
        batch_frames, _, channels = sparse_tokens.shape
        patch_height, patch_width = patch_hw
        coarse_height = max(1, math.ceil(patch_height / self.sparse_coarse_stride))
        coarse_width = max(1, math.ceil(patch_width / self.sparse_coarse_stride))

        coord_y = torch.clamp(
            torch.div(sparse_coords[..., 0], self.sparse_coarse_stride, rounding_mode="floor").long(),
            min=0,
            max=coarse_height - 1,
        )
        coord_x = torch.clamp(
            torch.div(sparse_coords[..., 1], self.sparse_coarse_stride, rounding_mode="floor").long(),
            min=0,
            max=coarse_width - 1,
        )
        flat_index = coord_y * coarse_width + coord_x

        accum = sparse_tokens.new_zeros(batch_frames, channels, coarse_height * coarse_width)
        norm = sparse_tokens.new_zeros(batch_frames, 1, coarse_height * coarse_width)
        weighted_tokens = sparse_tokens.transpose(1, 2) * sparse_sizes.unsqueeze(1).to(sparse_tokens.dtype)
        accum.scatter_add_(2, flat_index.unsqueeze(1).expand(-1, channels, -1), weighted_tokens)
        norm.scatter_add_(2, flat_index.unsqueeze(1), sparse_sizes.unsqueeze(1).to(sparse_tokens.dtype))
        coarse_map = accum / norm.clamp_min(1e-6)
        return coarse_map.reshape(batch_frames, channels, coarse_height, coarse_width)

    def forward(
        self,
        dense_feature_maps: list[torch.Tensor],
        sparse_patch_tokens: torch.Tensor,
        sparse_patch_coords: torch.Tensor,
        sparse_token_sizes: torch.Tensor,
    ) -> torch.Tensor:
        if len(dense_feature_maps) != self.num_dense_sources:
            raise ValueError(
                f"decoder_simple expects {self.num_dense_sources} dense feature maps, got {len(dense_feature_maps)}"
            )

        detail_map = dense_feature_maps[-1]
        if detail_map.ndim != 5:
            raise ValueError("decoder_simple expects dense feature maps with shape [B, T, C, H, W]")
        if self.num_dense_sources == 2:
            skip_map = dense_feature_maps[0]
            if skip_map.ndim != 5:
                raise ValueError("decoder_simple expects dense feature maps with shape [B, T, C, H, W]")
        else:
            skip_map = None

        batch_size, num_frames, channels, patch_height, patch_width = detail_map.shape
        batch_frames = batch_size * num_frames

        detail_flat = self.detail_proj(detail_map.reshape(batch_frames, channels, patch_height, patch_width))
        dense_inputs = [detail_flat]
        if skip_map is not None and self.skip_proj is not None:
            dense_inputs.insert(0, self.skip_proj(skip_map.reshape(batch_frames, channels, patch_height, patch_width)))

        sparse_projected = self.sparse_proj(sparse_patch_tokens)
        sparse_coarse = self._rasterize_sparse_tokens(
            sparse_tokens=sparse_projected,
            sparse_coords=sparse_patch_coords,
            sparse_sizes=sparse_token_sizes,
            patch_hw=(patch_height, patch_width),
        )
        sparse_coarse = sparse_coarse.reshape(
            batch_size,
            num_frames,
            sparse_coarse.shape[1],
            sparse_coarse.shape[2],
            sparse_coarse.shape[3],
        )
        sparse_fused = self.sparse_fusion([sparse_coarse])[0]
        sparse_fused = sparse_fused.reshape(
            batch_frames,
            sparse_fused.shape[2],
            sparse_fused.shape[3],
            sparse_fused.shape[4],
        )
        sparse_fused = F.interpolate(
            sparse_fused,
            size=(patch_height, patch_width),
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat((*dense_inputs, sparse_fused), dim=1)
        fused = self.fuse(fused)
        fused = self.refine(fused)
        return self.output_upsample(fused)
