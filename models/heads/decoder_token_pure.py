import torch
import torch.nn.functional as F
from torch import nn

from rvsd.ops import group_restore_to_map

from .dpt_head import ConvNormAct


class SparseTokenTemporalMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, tokens: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        if num_frames <= 1:
            return tokens
        batch_frames, num_tokens, channels = tokens.shape
        normed = self.norm(tokens)
        sequence = normed.reshape(batch_size, num_frames, num_tokens, channels)
        sequence = sequence.permute(0, 2, 3, 1).reshape(batch_size * num_tokens, channels, num_frames)
        mixed = self.pointwise(self.act(self.depthwise(sequence)))
        mixed = mixed.reshape(batch_size, num_tokens, channels, num_frames).permute(0, 3, 1, 2)
        return tokens + mixed.reshape(batch_frames, num_tokens, channels)


class DecoderTokenPureHead(nn.Module):
    """
    Sparse-token decoder with late dense restoration.

    Main computation stays on the sparse token sequence. The decoder restores to
    the dense patch grid only once near the end, then applies a lightweight
    dense refinement head for mask prediction.
    """

    def __init__(
        self,
        in_channels: int,
        decoder_dim: int = 192,
        output_dim: int = 128,
        use_batchnorm: bool = True,
        sparse_coarse_stride: int = 2,
        sparse_temporal_layers: int = 1,
        use_dark_guidance: bool = True,
        use_structure_decoupling: bool = True,
        use_structure_aux: bool = True,
    ):
        super().__init__()
        if sparse_coarse_stride < 1:
            raise ValueError(f"sparse_coarse_stride must be >= 1, got {sparse_coarse_stride}")
        self.sparse_coarse_stride = sparse_coarse_stride
        self.use_dark_guidance = bool(use_dark_guidance)
        self.use_structure_decoupling = bool(use_structure_decoupling)
        self.use_structure_aux = bool(use_structure_aux)

        self.token_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, decoder_dim),
            nn.GELU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.size_proj = nn.Sequential(
            nn.Linear(1, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.token_refine = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_dim * 2, decoder_dim),
        )
        self.token_mixers = nn.ModuleList(
            [SparseTokenTemporalMixer(decoder_dim) for _ in range(max(int(sparse_temporal_layers), 0))]
        )
        self.dark_token_proj = (
            nn.Sequential(
                nn.Linear(1, decoder_dim),
                nn.GELU(),
                nn.Linear(decoder_dim, decoder_dim),
            )
            if self.use_dark_guidance
            else None
        )
        self.dark_guidance_proj = (
            nn.Sequential(
                ConvNormAct(1, decoder_dim, use_batchnorm=use_batchnorm),
                ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            )
            if self.use_dark_guidance
            else None
        )
        self.restore_fuse = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
        )
        self.coarse_context = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
        )
        self.body_branch = (
            nn.Sequential(
                ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
                ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            )
            if self.use_structure_decoupling
            else None
        )
        self.detail_branch = (
            nn.Sequential(
                ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
                ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            )
            if self.use_structure_decoupling
            else None
        )
        self.structure_fuse = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
        )
        self.refine = nn.Sequential(
            ConvNormAct(decoder_dim, decoder_dim, use_batchnorm=use_batchnorm),
            ConvNormAct(decoder_dim, output_dim, use_batchnorm=use_batchnorm),
        )
        self.output_upsample = nn.ConvTranspose2d(output_dim, output_dim, kernel_size=4, stride=4)

        if self.use_structure_aux:
            self.body_aux_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
            self.detail_aux_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
            self.boundary_aux_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        else:
            self.body_aux_head = None
            self.detail_aux_head = None
            self.boundary_aux_head = None

    def _sample_dark_prior_at_tokens(
        self,
        dark_prior_map: torch.Tensor | None,
        sparse_patch_coords: torch.Tensor,
        patch_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        if dark_prior_map is None or self.dark_token_proj is None:
            return None
        patch_height, patch_width = patch_hw
        coord_y = torch.round(sparse_patch_coords[..., 0] - 0.5).long().clamp(0, patch_height - 1)
        coord_x = torch.round(sparse_patch_coords[..., 1] - 0.5).long().clamp(0, patch_width - 1)
        flat_index = (coord_y * patch_width + coord_x).unsqueeze(1)
        flat_prior = dark_prior_map.reshape(dark_prior_map.shape[0], 1, patch_height * patch_width)
        return flat_prior.gather(2, flat_index).transpose(1, 2)

    def _encode_sparse_tokens(
        self,
        sparse_patch_tokens: torch.Tensor,
        sparse_patch_coords: torch.Tensor,
        sparse_token_sizes: torch.Tensor,
        patch_hw: tuple[int, int],
        dark_prior_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        patch_height, patch_width = patch_hw
        coords = sparse_patch_coords.to(sparse_patch_tokens.dtype)
        coords_y = 2.0 * (coords[..., 0] / max(patch_height, 1)) - 1.0
        coords_x = 2.0 * (coords[..., 1] / max(patch_width, 1)) - 1.0
        coords_norm = torch.stack((coords_y, coords_x), dim=-1)
        size_feat = torch.log1p(sparse_token_sizes.to(sparse_patch_tokens.dtype)).unsqueeze(-1)

        token_features = self.token_proj(sparse_patch_tokens)
        token_features = token_features + self.coord_proj(coords_norm) + self.size_proj(size_feat)
        if dark_prior_tokens is not None and self.dark_token_proj is not None:
            token_features = token_features + self.dark_token_proj(dark_prior_tokens.to(token_features.dtype))
        token_features = token_features + self.token_refine(token_features)
        return token_features

    def _mix_sparse_tokens(
        self,
        sparse_tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        mixed = sparse_tokens
        for mixer in self.token_mixers:
            mixed = mixer(mixed, batch_size=batch_size, num_frames=num_frames)
        return mixed

    def _restore_sparse_tokens_to_map(
        self,
        sparse_tokens: torch.Tensor,
        restore_index: torch.Tensor | None,
        patch_hw: tuple[int, int],
    ) -> torch.Tensor:
        patch_height, patch_width = patch_hw
        if restore_index is None:
            if sparse_tokens.shape[1] != patch_height * patch_width:
                raise ValueError(
                    "restore_index is required when sparse token count does not match the dense patch grid size."
                )
            dense_tokens = sparse_tokens
            return dense_tokens.reshape(
                dense_tokens.shape[0],
                patch_height,
                patch_width,
                dense_tokens.shape[-1],
            ).permute(0, 3, 1, 2).contiguous()
        return group_restore_to_map(sparse_tokens, restore_index, patch_height, patch_width)

    def _apply_dense_context(
        self,
        dense_patch_map: torch.Tensor,
    ) -> torch.Tensor:
        refined = self.restore_fuse(dense_patch_map)
        if self.sparse_coarse_stride <= 1:
            return refined
        coarse = F.avg_pool2d(
            refined,
            kernel_size=self.sparse_coarse_stride,
            stride=self.sparse_coarse_stride,
        )
        coarse = self.coarse_context(coarse)
        coarse = F.interpolate(coarse, size=refined.shape[-2:], mode="bilinear", align_corners=False)
        return refined + coarse

    def _apply_structure_decoupling(
        self,
        dense_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smooth_context = F.avg_pool2d(dense_feature, kernel_size=3, stride=1, padding=1)
        if self.body_branch is None or self.detail_branch is None:
            body_feature = smooth_context
            detail_feature = dense_feature - smooth_context
        else:
            body_feature = self.body_branch(smooth_context)
            detail_feature = self.detail_branch(dense_feature - smooth_context)
        structure_feature = self.structure_fuse(dense_feature + body_feature + detail_feature)
        return structure_feature, body_feature, detail_feature

    def forward(
        self,
        sparse_patch_tokens: torch.Tensor,
        sparse_patch_coords: torch.Tensor,
        sparse_token_sizes: torch.Tensor,
        restore_index: torch.Tensor | None,
        patch_hw: tuple[int, int],
        batch_size: int,
        num_frames: int,
        dark_prior_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        dark_prior_tokens = self._sample_dark_prior_at_tokens(
            dark_prior_map=dark_prior_map,
            sparse_patch_coords=sparse_patch_coords,
            patch_hw=patch_hw,
        )
        sparse_projected = self._encode_sparse_tokens(
            sparse_patch_tokens=sparse_patch_tokens,
            sparse_patch_coords=sparse_patch_coords,
            sparse_token_sizes=sparse_token_sizes,
            patch_hw=patch_hw,
            dark_prior_tokens=dark_prior_tokens,
        )
        sparse_projected = self._mix_sparse_tokens(
            sparse_tokens=sparse_projected,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        dense_patch_map = self._restore_sparse_tokens_to_map(
            sparse_tokens=sparse_projected,
            restore_index=restore_index,
            patch_hw=patch_hw,
        )

        if self.dark_guidance_proj is not None and dark_prior_map is not None:
            dense_patch_map = dense_patch_map + self.dark_guidance_proj(dark_prior_map.to(dense_patch_map.dtype))

        dense_feature = self._apply_dense_context(dense_patch_map)
        structure_feature, body_feature, detail_feature = self._apply_structure_decoupling(dense_feature)
        refined_feature = self.refine(structure_feature)
        output_feature = self.output_upsample(refined_feature)

        auxiliary_outputs: dict[str, torch.Tensor] = {}
        if self.use_structure_aux:
            aux_target_size = output_feature.shape[-2:]
            body_logits = F.interpolate(
                self.body_aux_head(body_feature),
                size=aux_target_size,
                mode="bilinear",
                align_corners=False,
            )
            detail_logits = F.interpolate(
                self.detail_aux_head(detail_feature),
                size=aux_target_size,
                mode="bilinear",
                align_corners=False,
            )
            boundary_logits = F.interpolate(
                self.boundary_aux_head(detail_feature.abs()),
                size=aux_target_size,
                mode="bilinear",
                align_corners=False,
            )
            auxiliary_outputs = {
                "body_logits": body_logits,
                "detail_logits": detail_logits,
                "boundary_logits": boundary_logits,
            }

        return {
            "features": output_feature,
            "auxiliary": auxiliary_outputs,
        }
