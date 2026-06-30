from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GroupLayout:
    group_member_indices: torch.Tensor
    group_member_mask: torch.Tensor
    group_centers: torch.Tensor
    patch_coords: torch.Tensor
    patch_height: int
    patch_width: int

    @property
    def num_groups(self) -> int:
        return int(self.group_member_indices.shape[0])

    @property
    def max_group_size(self) -> int:
        return int(self.group_member_indices.shape[1])


def _token_grid_indices(token_coords: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    h_index = torch.round(token_coords[:, 0] - 0.5).long().clamp(0, patch_height - 1)
    w_index = torch.round(token_coords[:, 1] - 0.5).long().clamp(0, patch_width - 1)
    return h_index * patch_width + w_index


def _scatter_tokens_to_grid(
    token_features: torch.Tensor,
    token_coords: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_features.ndim != 2:
        raise ValueError(f"Expected token_features [N, C], got {tuple(token_features.shape)}")

    num_tokens, channels = token_features.shape
    flat_index = _token_grid_indices(token_coords, patch_height, patch_width)
    grid = token_features.new_zeros(1, channels, patch_height * patch_width)
    counts = token_features.new_zeros(1, 1, patch_height * patch_width)

    scatter_index = flat_index.unsqueeze(0).expand(channels, -1).unsqueeze(0)
    grid.scatter_add_(2, scatter_index, token_features.transpose(0, 1).unsqueeze(0))
    counts.scatter_add_(2, flat_index.view(1, 1, num_tokens), token_features.new_ones(1, 1, num_tokens))
    grid = grid / counts.clamp_min(1.0)
    return grid.reshape(1, channels, patch_height, patch_width), flat_index


def build_group_layout(patch_height: int, patch_width: int, group_size: tuple[int, int]) -> GroupLayout:
    group_height, group_width = group_size
    if group_height < 1 or group_width < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    h_coords = torch.arange(patch_height, dtype=torch.float32) + 0.5
    w_coords = torch.arange(patch_width, dtype=torch.float32) + 0.5
    patch_coords = torch.stack(torch.meshgrid(h_coords, w_coords, indexing="ij"), dim=-1).reshape(-1, 2)

    grid_rows = (patch_height + group_height - 1) // group_height
    grid_cols = (patch_width + group_width - 1) // group_width
    padded_height = grid_rows * group_height
    padded_width = grid_cols * group_width

    padded_indices = torch.full((padded_height, padded_width), -1, dtype=torch.long)
    flat_indices = torch.arange(patch_height * patch_width, dtype=torch.long).reshape(patch_height, patch_width)
    padded_indices[:patch_height, :patch_width] = flat_indices

    member_indices = (
        padded_indices
        .reshape(grid_rows, group_height, grid_cols, group_width)
        .permute(0, 2, 1, 3)
        .reshape(grid_rows * grid_cols, group_height * group_width)
    )
    member_mask = member_indices >= 0
    member_offsets = torch.arange(member_indices.shape[1], dtype=torch.long).view(1, -1).expand_as(member_indices)
    member_order = (member_mask.logical_not().long() * member_indices.shape[1] + member_offsets).argsort(dim=1)
    member_indices = member_indices.gather(1, member_order)
    member_mask = member_mask.gather(1, member_order)
    member_indices_clamped = member_indices.clamp_min(0)
    member_counts = member_mask.sum(dim=1).clamp_min(1)
    grouped_coords = patch_coords[member_indices_clamped]
    group_centers = (grouped_coords * member_mask.unsqueeze(-1).to(grouped_coords.dtype)).sum(dim=1)
    group_centers = group_centers / member_counts.unsqueeze(-1).to(grouped_coords.dtype)

    return GroupLayout(
        group_member_indices=member_indices,
        group_member_mask=member_mask,
        group_centers=group_centers.float(),
        patch_coords=patch_coords.float(),
        patch_height=patch_height,
        patch_width=patch_width,
    )


def build_dynamic_group_layout(
    token_coords: torch.Tensor,
    patch_height: int,
    patch_width: int,
    group_size: tuple[int, int],
) -> GroupLayout:
    group_height, group_width = group_size
    bin_to_members: dict[tuple[int, int], list[int]] = {}
    coords_cpu = token_coords.detach().cpu()

    for token_index, coord in enumerate(coords_cpu):
        h_bin = min(max(int((float(coord[0]) - 0.5) // group_height), 0), max((patch_height - 1) // group_height, 0))
        w_bin = min(max(int((float(coord[1]) - 0.5) // group_width), 0), max((patch_width - 1) // group_width, 0))
        bin_to_members.setdefault((h_bin, w_bin), []).append(token_index)

    ordered_bins = sorted(bin_to_members.keys())
    group_members = [bin_to_members[key] for key in ordered_bins]
    group_centers = []
    for members in group_members:
        group_centers.append(token_coords[members].mean(dim=0))

    return build_layout_from_members(
        group_members=group_members,
        patch_coords=token_coords.detach().cpu(),
        patch_height=patch_height,
        patch_width=patch_width,
        group_centers=torch.stack(group_centers, dim=0).cpu(),
    )


def build_layout_from_members(
    group_members: list[list[int]],
    patch_coords: torch.Tensor,
    patch_height: int,
    patch_width: int,
    group_centers: torch.Tensor | None = None,
) -> GroupLayout:
    max_group_size = max(len(members) for members in group_members)
    member_indices = torch.full((len(group_members), max_group_size), -1, dtype=torch.long)
    member_mask = torch.zeros((len(group_members), max_group_size), dtype=torch.bool)
    if group_centers is None:
        centers = []
    else:
        centers = None

    for group_index, members in enumerate(group_members):
        member_indices[group_index, : len(members)] = torch.tensor(members, dtype=torch.long)
        member_mask[group_index, : len(members)] = True
        if centers is not None:
            centers.append(patch_coords[members].float().mean(dim=0))

    if centers is not None:
        group_centers = torch.stack(centers, dim=0)

    return GroupLayout(
        group_member_indices=member_indices,
        group_member_mask=member_mask,
        group_centers=group_centers.float(),
        patch_coords=patch_coords.float(),
        patch_height=patch_height,
        patch_width=patch_width,
    )


def compute_group_scores(token_scores: torch.Tensor, layout: GroupLayout) -> torch.Tensor:
    if token_scores.ndim != 2:
        raise ValueError(f"Expected token_scores [B, N], got {tuple(token_scores.shape)}")
    gather_indices = layout.group_member_indices.to(token_scores.device)
    gather_mask = layout.group_member_mask.to(token_scores.device)
    expanded = token_scores[:, gather_indices.clamp_min(0)]
    expanded = expanded.masked_fill(~gather_mask.unsqueeze(0), 0.0)
    denom = gather_mask.sum(dim=1).clamp_min(1).unsqueeze(0).to(token_scores.dtype)
    return expanded.sum(dim=-1) / denom


class _SpatialTokenHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        token_coords: torch.Tensor,
    ) -> torch.Tensor:
        patch_height, patch_width = patch_hw
        token_features = self.pre(patch_tokens)
        token_grid, flat_index = _scatter_tokens_to_grid(
            token_features=token_features,
            token_coords=token_coords,
            patch_height=patch_height,
            patch_width=patch_width,
        )
        token_scores_grid = self.spatial(token_grid).reshape(-1)
        return token_scores_grid[flat_index]


class TokenDifficultyHead(_SpatialTokenHead):
    pass


class TokenShadowHead(_SpatialTokenHead):
    pass


class _TokenMLPHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_hw: tuple[int, int] | None = None,
        token_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del patch_hw, token_coords
        return self.net(patch_tokens).squeeze(-1)


class TokenDifficultyMLPHead(_TokenMLPHead):
    pass


class TokenShadowMLPHead(_TokenMLPHead):
    pass


class GroupRiskHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_tokens: torch.Tensor, layout: GroupLayout) -> torch.Tensor:
        member_indices = layout.group_member_indices.to(patch_tokens.device)
        member_mask = layout.group_member_mask.to(patch_tokens.device)

        if patch_tokens.ndim == 2:
            gathered = patch_tokens[member_indices.clamp_min(0)]
            gathered = gathered * member_mask.unsqueeze(-1).to(gathered.dtype)
            pooled = gathered.sum(dim=1) / member_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).to(gathered.dtype)
            return self.net(pooled).squeeze(-1)

        if patch_tokens.ndim == 3:
            batch_size, _num_tokens, channels = patch_tokens.shape
            num_groups, max_group_size = member_indices.shape
            gather_index = member_indices.view(1, num_groups, max_group_size, 1).expand(batch_size, -1, -1, channels)
            gathered = patch_tokens.gather(
                1,
                gather_index.reshape(batch_size, num_groups * max_group_size, channels),
            ).reshape(batch_size, num_groups, max_group_size, channels)
            gathered = gathered * member_mask.view(1, num_groups, max_group_size, 1).to(gathered.dtype)
            denom = member_mask.sum(dim=1).clamp_min(1).view(1, num_groups, 1).to(gathered.dtype)
            pooled = gathered.sum(dim=2) / denom
            return self.net(pooled).squeeze(-1)

        raise ValueError(f"Expected patch_tokens [N, C] or [B, N, C], got {tuple(patch_tokens.shape)}")


class _GroupStatHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, stats_dim: int):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, pooled_tokens: torch.Tensor, group_stats: torch.Tensor) -> torch.Tensor:
        hidden = self.feature_proj(pooled_tokens) + self.stats_proj(group_stats)
        return self.output(hidden).squeeze(-1)


class GroupCompressibilityHead(_GroupStatHead):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__(embed_dim=embed_dim, hidden_dim=hidden_dim, stats_dim=2)


class GroupTaskAwarePreserveHead(_GroupStatHead):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__(embed_dim=embed_dim, hidden_dim=hidden_dim, stats_dim=4)


class TokenTransformReducer(nn.Module):
    """
    Merge-time token reducer inspired by token transforming.

    Instead of emitting a plain weighted mean token for a merged group, this
    module lets the model learn a small residual transform conditioned on
    pooled content, relative geometry and aggregate token area.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        coord_scale: float = 1.0,
        size_scale: float = 1.0,
    ):
        super().__init__()
        self.coord_scale = float(coord_scale)
        self.size_scale = float(size_scale)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.size_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        grouped_tokens: torch.Tensor,
        grouped_coords: torch.Tensor,
        grouped_sizes: torch.Tensor | None,
        member_mask: torch.Tensor,
        patch_hw: tuple[int, int],
    ) -> torch.Tensor:
        if grouped_tokens.ndim != 4:
            raise ValueError(f"Expected grouped_tokens [B, G, K, C], got {tuple(grouped_tokens.shape)}")

        patch_height, patch_width = patch_hw
        dtype = grouped_tokens.dtype
        device = grouped_tokens.device
        if member_mask.ndim == 2:
            member_mask_batch = member_mask.unsqueeze(0).expand(grouped_tokens.shape[0], -1, -1)
        elif member_mask.ndim == 3:
            member_mask_batch = member_mask
        else:
            raise ValueError(f"Expected member_mask [G, K] or [B, G, K], got {tuple(member_mask.shape)}")
        member_mask_batch = member_mask_batch.to(device=device)
        member_mask_float = member_mask_batch.to(dtype)

        if grouped_sizes is None:
            grouped_sizes = grouped_tokens.new_ones(grouped_tokens.shape[:3])
        grouped_sizes = grouped_sizes.to(dtype) * member_mask_float
        weight_denom = grouped_sizes.sum(dim=2, keepdim=True).clamp_min(1.0)

        pooled_tokens = (grouped_tokens * grouped_sizes.unsqueeze(-1)).sum(dim=2) / weight_denom
        pooled_coords = (grouped_coords * grouped_sizes.unsqueeze(-1).to(grouped_coords.dtype)).sum(dim=2) / weight_denom

        rel_coords = grouped_coords - pooled_coords.unsqueeze(2)
        rel_coords = rel_coords.to(dtype)
        rel_coords = torch.stack(
            (
                rel_coords[..., 0] / max(float(patch_height), 1.0),
                rel_coords[..., 1] / max(float(patch_width), 1.0),
            ),
            dim=-1,
        )
        size_feat = torch.log1p(grouped_sizes.clamp_min(1.0)).unsqueeze(-1)

        hidden = self.token_proj(grouped_tokens)
        hidden = hidden + self.coord_scale * self.coord_proj(rel_coords) + self.size_scale * self.size_proj(size_feat)
        hidden = hidden * member_mask_float.unsqueeze(-1)
        pooled_hidden = (hidden * grouped_sizes.unsqueeze(-1)).sum(dim=2) / weight_denom

        avg_extent = (rel_coords.abs() * grouped_sizes.unsqueeze(-1)).sum(dim=2) / weight_denom
        log_area = torch.log1p(weight_denom.squeeze(-1)).unsqueeze(-1)
        group_stats = torch.cat((avg_extent, log_area), dim=-1)
        fused = pooled_hidden + self.stats_proj(group_stats)
        return pooled_tokens + self.output(fused)


class TemporalConsensusCorrection(nn.Module):
    """
    Lightweight temporal consensus refinement on sparse tokens.

    Tokens first look up a coarse spatio-temporal consensus feature using their
    spatial bin across nearby frames, then predict a residual correction.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.consensus_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.size_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_coords: torch.Tensor,
        token_sizes: torch.Tensor | None,
        patch_hw: tuple[int, int],
        batch_size: int,
        num_frames: int,
        stride: int = 4,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch_tokens [B*T, N, C], got {tuple(patch_tokens.shape)}")
        if num_frames <= 1:
            return patch_tokens

        batch_frames, num_tokens, channels = patch_tokens.shape
        if batch_frames != batch_size * num_frames:
            raise ValueError(
                f"Expected batch_frames == batch_size * num_frames, got {batch_frames} vs {batch_size * num_frames}"
            )
        patch_height, patch_width = patch_hw
        stride = max(int(stride), 1)
        bin_height = max(1, (patch_height + stride - 1) // stride)
        bin_width = max(1, (patch_width + stride - 1) // stride)
        num_bins = bin_height * bin_width

        tokens_bt = patch_tokens.reshape(batch_size, num_frames, num_tokens, channels)
        coords_bt = patch_coords.reshape(batch_size, num_frames, num_tokens, 2).to(patch_tokens.dtype)
        if token_sizes is None:
            weights_bt = patch_tokens.new_ones((batch_size, num_frames, num_tokens))
        else:
            weights_bt = token_sizes.reshape(batch_size, num_frames, num_tokens).to(patch_tokens.dtype)

        y_bin = ((coords_bt[..., 0] - 0.5) / float(stride)).floor().long().clamp(0, bin_height - 1)
        x_bin = ((coords_bt[..., 1] - 0.5) / float(stride)).floor().long().clamp(0, bin_width - 1)
        bin_index = y_bin * bin_width + x_bin

        flat_tokens = tokens_bt.reshape(batch_size, num_frames * num_tokens, channels)
        flat_bins = bin_index.reshape(batch_size, num_frames * num_tokens)
        flat_weights = weights_bt.reshape(batch_size, num_frames * num_tokens)

        accum = patch_tokens.new_zeros(batch_size, num_bins, channels)
        counts = patch_tokens.new_zeros(batch_size, num_bins, 1)
        accum.scatter_add_(
            1,
            flat_bins.unsqueeze(-1).expand(-1, -1, channels),
            flat_tokens * flat_weights.unsqueeze(-1),
        )
        counts.scatter_add_(1, flat_bins.unsqueeze(-1), flat_weights.unsqueeze(-1))
        consensus = accum / counts.clamp_min(1.0)
        gathered_consensus = consensus.gather(1, flat_bins.unsqueeze(-1).expand(-1, -1, channels)).reshape(
            batch_size,
            num_frames,
            num_tokens,
            channels,
        )

        coords_norm = torch.stack(
            (
                2.0 * (coords_bt[..., 0] / max(float(patch_height), 1.0)) - 1.0,
                2.0 * (coords_bt[..., 1] / max(float(patch_width), 1.0)) - 1.0,
            ),
            dim=-1,
        )
        size_feat = torch.log1p(weights_bt.clamp_min(1.0)).unsqueeze(-1)
        hidden = self.token_proj(tokens_bt)
        hidden = hidden + self.consensus_proj(gathered_consensus)
        hidden = hidden + self.coord_proj(coords_norm) + self.size_proj(size_feat)
        corrected = tokens_bt + self.output(hidden)
        return corrected.reshape(batch_frames, num_tokens, channels)


class BudgetHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_tokens).squeeze(-1)


class SizeCorrection(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(1, embed_dim)
        self.bias_proj = nn.Linear(1, embed_dim)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.bias_proj.weight)
        nn.init.zeros_(self.bias_proj.bias)

    def forward(self, token_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_size = torch.log(token_sizes.clamp_min(1.0)).unsqueeze(-1)
        scale = 1.0 + 0.25 * torch.tanh(self.scale_proj(log_size))
        bias = self.bias_proj(log_size)
        return scale, bias


class MergeCorrection(nn.Module):
    """
    Lightweight post-merge correction module.

    It predicts:
    - a residual feature correction for merged patch tokens
    - a learned attention-bias term for later transformer blocks
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.source_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.variance_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.size_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.bias_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.bias_head[-1].weight)
        nn.init.zeros_(self.bias_head[-1].bias)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_coords: torch.Tensor,
        token_sizes: torch.Tensor,
        patch_hw: tuple[int, int],
        source_tokens: torch.Tensor | None = None,
        source_coords: torch.Tensor | None = None,
        source_sizes: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_height, patch_width = patch_hw
        coords = patch_coords.to(patch_tokens.dtype)
        coords_norm = torch.stack(
            (
                2.0 * (coords[..., 0] / max(patch_height, 1)) - 1.0,
                2.0 * (coords[..., 1] / max(patch_width, 1)) - 1.0,
            ),
            dim=-1,
        )
        size_feat = torch.log1p(token_sizes.clamp_min(1.0)).unsqueeze(-1).to(patch_tokens.dtype)
        hidden = self.token_proj(patch_tokens)
        hidden = hidden + self.coord_proj(coords_norm) + self.size_proj(size_feat)

        if source_tokens is not None:
            if source_mask is None:
                source_mask = torch.ones(source_tokens.shape[:3], device=source_tokens.device, dtype=torch.bool)
            source_mask = source_mask.to(device=source_tokens.device)
            source_weights = source_mask.to(source_tokens.dtype)
            if source_sizes is not None:
                source_weights = source_weights * source_sizes.to(source_tokens.dtype)
            weight_denom = source_weights.sum(dim=2, keepdim=True).clamp_min(1.0)

            pooled_source = (source_tokens * source_weights.unsqueeze(-1)).sum(dim=2) / weight_denom
            centered_source = source_tokens - pooled_source.unsqueeze(2)
            source_variance = (centered_source.pow(2) * source_weights.unsqueeze(-1)).sum(dim=2) / weight_denom

            if source_coords is not None:
                weighted_coords = source_coords.to(source_tokens.dtype)
                pooled_coords = (weighted_coords * source_weights.unsqueeze(-1)).sum(dim=2) / weight_denom
                rel_coords = weighted_coords - pooled_coords.unsqueeze(2)
                avg_extent = (rel_coords.abs() * source_weights.unsqueeze(-1)).sum(dim=2) / weight_denom
                avg_extent = torch.stack(
                    (
                        avg_extent[..., 0] / max(float(patch_height), 1.0),
                        avg_extent[..., 1] / max(float(patch_width), 1.0),
                    ),
                    dim=-1,
                )
            else:
                avg_extent = patch_tokens.new_zeros((patch_tokens.shape[0], patch_tokens.shape[1], 2))

            mean_std = source_variance.mean(dim=-1, keepdim=True).sqrt()
            log_area = torch.log1p(weight_denom.squeeze(-1)).unsqueeze(-1)
            group_stats = torch.cat((avg_extent, mean_std, log_area), dim=-1)

            hidden = hidden + self.source_proj(pooled_source)
            hidden = hidden + self.variance_proj(source_variance.sqrt())
            hidden = hidden + self.stats_proj(group_stats)

        token_residual = self.residual_head(hidden)
        attn_bias = self.bias_head(hidden).squeeze(-1)
        return token_residual, attn_bias


class ExistenceHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_clip_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_clip_tokens).squeeze(-1)
