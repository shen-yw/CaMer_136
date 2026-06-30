from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import nn

_LOCAL_DINOV3_REPO = Path(__file__).resolve().parents[3] / "dinov3"
if _LOCAL_DINOV3_REPO.exists():
    local_dinov3_path = str(_LOCAL_DINOV3_REPO)
    if local_dinov3_path not in sys.path:
        sys.path.insert(0, local_dinov3_path)

from dinov3.hub.backbones import dinov3_vitb16
from rvsd.configs.baseline import AdaptiveComputeConfig
from rvsd.models.backbones.adaptive_compute import (
    ExistenceHead,
    GroupCompressibilityHead,
    GroupRiskHead,
    GroupTaskAwarePreserveHead,
    GroupLayout,
    MergeCorrection,
    SizeCorrection,
    TemporalConsensusCorrection,
    TokenShadowMLPHead,
    TokenTransformReducer,
    build_group_layout,
    compute_group_scores,
)
from rvsd.ops import (
    group_contract_available,
    group_contract_merge,
    group_contract_merge_mask_unweighted_with_sizes,
    group_contract_merge_unweighted_with_sizes,
    group_contract_plan,
    group_restore,
    group_restore_to_map,
)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_local_weights_path(weights_path: str) -> str:
    candidate = Path(weights_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    search_roots = [Path.cwd(), _workspace_root()]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str((_workspace_root() / candidate).resolve())


@dataclass
class BatchMergePlan:
    dst_index: torch.Tensor
    old_to_new: torch.Tensor
    out_rows: int
    num_merge_groups: int


class DinoV3ViTBWrapper(nn.Module):
    """
    Task-local wrapper around DINOv3 ViT-B for dense video prediction.

    Token merging, sparse-tail execution, and all task-specific compute control
    live here only. Core DINOv3 transformer internals remain untouched.
    """

    def __init__(
        self,
        weights_path: str,
        out_indices: tuple[int, int, int, int] = (2, 5, 8, 11),
        use_backbone_norm: bool = True,
        train_policy: str = "last_n_blocks",
        train_last_n_blocks: int = 4,
        adaptive_config: AdaptiveComputeConfig | None = None,
    ):
        super().__init__()
        self.weights_path = resolve_local_weights_path(weights_path)
        self.out_indices = tuple(out_indices)
        self.use_backbone_norm = use_backbone_norm
        self.adaptive_config = adaptive_config or AdaptiveComputeConfig()

        self.backbone = dinov3_vitb16(pretrained=True, weights=self.weights_path)
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        self.n_blocks = len(self.backbone.blocks)
        self.token_prefix = 1 + self.backbone.n_storage_tokens
        self._has_trainable_backbone = False
        self.runtime_mode = self.adaptive_config.runtime_mode.lower()
        if self.runtime_mode not in {"accuracy", "risk_v1"}:
            raise ValueError(f"Unsupported adaptive runtime mode: {self.adaptive_config.runtime_mode}")
        self.use_risk_v1_mode = (
            self.adaptive_config.enable_token_merging
            and self.runtime_mode == "risk_v1"
        )

        self.train_policy = ""
        self.train_last_n_blocks = 0
        self.apply_training_policy(train_policy=train_policy, train_last_n_blocks=train_last_n_blocks)

        self.group_risk_head = (
            GroupRiskHead(self.embed_dim, self.adaptive_config.risk_hidden_dim)
            if self.use_risk_v1_mode
            else None
        )
        self._risk_head_eager_net: nn.Module | None = self.group_risk_head.net if self.group_risk_head is not None else None
        self._risk_head_compiled = False
        self.group_compressibility_head = (
            GroupCompressibilityHead(self.embed_dim, self.adaptive_config.compressibility_hidden_dim)
            if self.use_risk_v1_mode and self.adaptive_config.use_independent_compressibility_predictor
            else None
        )
        self.group_task_preserve_head = (
            GroupTaskAwarePreserveHead(self.embed_dim, self.adaptive_config.task_preserve_hidden_dim)
            if self.use_risk_v1_mode and self.adaptive_config.use_task_aware_preservation
            else None
        )
        self.task_token_shadow_head = (
            TokenShadowMLPHead(self.embed_dim, self.adaptive_config.token_label_hidden_dim)
            if self.use_risk_v1_mode and self.adaptive_config.use_task_aware_preservation
            else None
        )
        if self.task_token_shadow_head is not None:
            nn.init.zeros_(self.task_token_shadow_head.net[-1].weight)
            nn.init.zeros_(self.task_token_shadow_head.net[-1].bias)
        self.size_correction = (
            SizeCorrection(self.embed_dim)
            if self.adaptive_config.enable_token_merging and self.adaptive_config.use_size_aware_correction
            else None
        )
        self.use_size_bias = bool(getattr(self.adaptive_config, "use_size_bias", True))
        self.merge_correction = (
            MergeCorrection(self.embed_dim, self.adaptive_config.merge_correction_hidden_dim)
            if self.adaptive_config.enable_token_merging and self.adaptive_config.use_merge_correction
            else None
        )
        self.token_transform_reducer = (
            TokenTransformReducer(
                self.embed_dim,
                self.adaptive_config.token_transform_hidden_dim,
                coord_scale=self.adaptive_config.token_transform_coord_scale,
                size_scale=self.adaptive_config.token_transform_size_scale,
            )
            if self.adaptive_config.enable_token_merging and self.adaptive_config.use_token_transform_reducer
            else None
        )
        self.temporal_consensus_correction = (
            TemporalConsensusCorrection(
                self.embed_dim,
                self.adaptive_config.temporal_consensus_hidden_dim,
            )
            if self.adaptive_config.use_temporal_consensus_correction
            else None
        )
        self.existence_head = (
            ExistenceHead(self.embed_dim, self.adaptive_config.existence_hidden_dim)
            if self.adaptive_config.enable_existence_calibration
            else None
        )
        self._group_layout_cache: dict[tuple[int, int, tuple[int, int]], GroupLayout] = {}
        self._group_layout_device_cache: dict[tuple[int, int, tuple[int, int], str], dict[str, torch.Tensor]] = {}
        self._base_rope_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        self._batch_index_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._runtime_arange_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._runtime_ones_cache: dict[tuple[tuple[int, ...], str, torch.dtype], torch.Tensor] = {}
        self._runtime_long_buffer_cache: dict[tuple[str, tuple[int, ...], str], torch.Tensor] = {}
        self._runtime_scatter_index_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self._group_contract_extension_enabled = bool(
            self.adaptive_config.use_group_contract_extension and group_contract_available()
        )
        self.decoder_variant = "decoder_token_pure"
        self.decoder_simple_num_dense_sources = 2
        self.enable_speed_profile = False
        self.enable_speed_profile_detail = False
        self.last_speed_profile: dict[str, float] = {}
        self._dynamic_keep_complexity_ema: float | None = None
        self._dynamic_keep_ratio_ema: float | None = None
        self._compiled_block_indices: tuple[int, ...] = tuple()

    def _uses_sparse_decoder(self) -> bool:
        return True

    def _decoder_requires_dense_sources(self) -> bool:
        return self.decoder_variant == "decoder_simple"

    def _size_bias_enabled(self) -> bool:
        return bool(self.use_size_bias and self.size_correction is not None)

    def _risk_head_scores(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        if self.group_risk_head is None:
            raise RuntimeError("risk_v1 requires a group risk head.")
        if (
            not self.training
            and not self._risk_head_compiled
            and bool(getattr(self.adaptive_config, "compile_risk_head", False))
            and hasattr(torch, "compile")
        ):
            self.group_risk_head.net = torch.compile(
                self.group_risk_head.net,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
            self._risk_head_compiled = True
        if self._risk_head_compiled:
            try:
                return self.group_risk_head.net(pooled_tokens).squeeze(-1)
            except Exception:
                if self._risk_head_eager_net is None:
                    raise
                self.group_risk_head.net = self._risk_head_eager_net
                self._risk_head_compiled = False
        return self.group_risk_head.net(pooled_tokens).squeeze(-1)

    def _profile_section_ms(self, device: torch.device, fn):
        if not self.enable_speed_profile:
            return fn(), 0.0
        start_marker = self._profile_marker(device)
        result = fn()
        end_marker = self._profile_marker(device)
        return result, self._profile_elapsed_ms(start_marker, end_marker, device)

    def _profile_marker(self, device: torch.device):
        if not self.enable_speed_profile:
            return None
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(device))
            return event
        return perf_counter()

    def _profile_elapsed_ms(self, start_marker, end_marker, device: torch.device) -> float:
        if start_marker is None or end_marker is None:
            return 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            return float(start_marker.elapsed_time(end_marker))
        return float((end_marker - start_marker) * 1000.0)

    def apply_training_policy(self, train_policy: str, train_last_n_blocks: int = 4) -> None:
        self.backbone.requires_grad_(False)
        self.train_policy = train_policy
        self.train_last_n_blocks = train_last_n_blocks

        if train_policy == "frozen":
            pass
        elif train_policy == "last_n_blocks":
            if train_last_n_blocks < 1 or train_last_n_blocks > self.n_blocks:
                raise ValueError(f"train_last_n_blocks must be within [1, {self.n_blocks}], got {train_last_n_blocks}")
            for block in self.backbone.blocks[-train_last_n_blocks:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)
            if self.backbone.cls_norm is not None:
                self.backbone.cls_norm.requires_grad_(True)
            if self.backbone.local_cls_norm is not None:
                self.backbone.local_cls_norm.requires_grad_(True)
        elif train_policy == "full":
            self.backbone.requires_grad_(True)
        else:
            raise ValueError(f"Unsupported backbone training policy: {train_policy}")
        self._has_trainable_backbone = any(parameter.requires_grad for parameter in self.backbone.parameters())

    @property
    def has_trainable_backbone(self) -> bool:
        return self._has_trainable_backbone

    def trainable_backbone_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def _group_layout(self, patch_height: int, patch_width: int) -> GroupLayout:
        cache_key = (patch_height, patch_width, tuple(self.adaptive_config.group_size))
        if cache_key not in self._group_layout_cache:
            self._group_layout_cache[cache_key] = build_group_layout(
                patch_height=patch_height,
                patch_width=patch_width,
                group_size=self.adaptive_config.group_size,
            )
        return self._group_layout_cache[cache_key]

    def _group_layout_tensors(self, layout: GroupLayout, device: torch.device) -> dict[str, torch.Tensor]:
        cache_key = (layout.patch_height, layout.patch_width, tuple(self.adaptive_config.group_size), str(device))
        if cache_key not in self._group_layout_device_cache:
            member_indices = layout.group_member_indices.to(device=device)
            member_mask = layout.group_member_mask.to(device=device)
            self._group_layout_device_cache[cache_key] = {
                "member_indices": member_indices,
                "member_indices_clamped": member_indices.clamp_min(0),
                "member_mask": member_mask,
                "member_counts": member_mask.sum(dim=1).clamp_min(1),
                "all_members_valid": bool(member_mask.all().item()),
                "member_offsets": torch.arange(member_indices.shape[1], device=device, dtype=torch.long).view(1, 1, -1),
                "group_centers": layout.group_centers.to(device=device),
                "patch_coords": layout.patch_coords.to(device=device),
                "patch_coords_by_group": layout.patch_coords.to(device=device)[member_indices.clamp_min(0)],
            }
        return self._group_layout_device_cache[cache_key]

    def _base_rope(self, patch_height: int, patch_width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (patch_height, patch_width, str(device))
        if cache_key not in self._base_rope_cache:
            self._base_rope_cache[cache_key] = self.backbone.rope_embed(H=patch_height, W=patch_width)
        return self._base_rope_cache[cache_key]

    def _batch_index_column(self, batch_size: int, device: torch.device) -> torch.Tensor:
        cache_key = (batch_size, str(device))
        if cache_key not in self._batch_index_cache:
            self._batch_index_cache[cache_key] = torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1)
        return self._batch_index_cache[cache_key]

    def _runtime_arange(self, length: int, device: torch.device) -> torch.Tensor:
        cache_key = (int(length), str(device))
        cached = self._runtime_arange_cache.get(cache_key)
        if cached is None:
            cached = torch.arange(length, device=device, dtype=torch.long)
            self._runtime_arange_cache[cache_key] = cached
        return cached

    def _runtime_ones(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (tuple(int(dim) for dim in shape), str(device), dtype)
        cached = self._runtime_ones_cache.get(cache_key)
        if cached is None:
            cached = torch.ones(shape, device=device, dtype=dtype)
            self._runtime_ones_cache[cache_key] = cached
        return cached

    def _runtime_long_buffer(self, name: str, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        cache_key = (name, tuple(int(dim) for dim in shape), str(device))
        cached = self._runtime_long_buffer_cache.get(cache_key)
        if cached is None:
            cached = torch.empty(shape, device=device, dtype=torch.long)
            self._runtime_long_buffer_cache[cache_key] = cached
        return cached

    def _runtime_member_scatter_index(
        self,
        member_indices: torch.Tensor,
        num_frames_total: int,
    ) -> torch.Tensor:
        cache_key = (int(num_frames_total), int(member_indices.data_ptr()), str(member_indices.device))
        cached = self._runtime_scatter_index_cache.get(cache_key)
        if cached is None:
            cached = member_indices.reshape(1, -1).expand(num_frames_total, -1)
            self._runtime_scatter_index_cache[cache_key] = cached
        return cached

    def compile_inference_blocks(
        self,
        block_indices: tuple[int, ...],
        *,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
    ) -> tuple[int, ...]:
        if not hasattr(torch, "compile"):
            return tuple()
        compiled_indices: list[int] = []
        for block_index in block_indices:
            block_index = int(block_index)
            if block_index < 0 or block_index >= self.n_blocks:
                continue
            if block_index in self._compiled_block_indices:
                compiled_indices.append(block_index)
                continue
            self.backbone.blocks[block_index] = torch.compile(
                self.backbone.blocks[block_index],
                mode=mode,
                fullgraph=bool(fullgraph),
                dynamic=False,
            )
            compiled_indices.append(block_index)
        self._compiled_block_indices = tuple(sorted(set(self._compiled_block_indices + tuple(compiled_indices))))
        return self._compiled_block_indices

    def _group_patch_tokens_and_coords(
        self,
        patch_tokens: torch.Tensor,
        layout: GroupLayout,
        layout_tensors: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch_tokens [B, N, C], got {tuple(patch_tokens.shape)}")
        cached = layout_tensors or self._group_layout_tensors(layout, patch_tokens.device)
        member_indices = cached["member_indices"]
        member_indices_clamped = cached["member_indices_clamped"]
        member_mask = cached["member_mask"]
        num_frames_total, _, channels = patch_tokens.shape
        num_groups, group_width = member_indices.shape
        gather_index = member_indices_clamped.view(1, num_groups, group_width, 1).expand(num_frames_total, -1, -1, channels)
        grouped_tokens = patch_tokens.gather(
            1,
            gather_index.reshape(num_frames_total, num_groups * group_width, channels),
        ).reshape(num_frames_total, num_groups, group_width, channels)
        grouped_coords = cached["patch_coords_by_group"].unsqueeze(0).expand(num_frames_total, -1, -1, -1)
        return grouped_tokens, grouped_coords, member_mask, cached["member_counts"]

    def _compute_group_scores_batch(
        self,
        token_scores: torch.Tensor,
        layout: GroupLayout,
        layout_tensors: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if token_scores.ndim != 2:
            raise ValueError(f"Expected token_scores [B, N], got {tuple(token_scores.shape)}")
        cached = layout_tensors or self._group_layout_tensors(layout, token_scores.device)
        member_indices = cached["member_indices"]
        member_mask = cached["member_mask"]
        expanded = token_scores[:, member_indices]
        expanded = expanded.masked_fill(~member_mask.unsqueeze(0), 0.0)
        denom = member_mask.sum(dim=1).clamp_min(1).unsqueeze(0).to(token_scores.dtype)
        return expanded.sum(dim=-1) / denom

    def _group_risk_scores_batch(
        self,
        patch_tokens: torch.Tensor,
        layout: GroupLayout,
        layout_tensors: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.group_risk_head is None:
            raise RuntimeError("risk_v1 requires a group risk head.")
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch_tokens [B, N, C], got {tuple(patch_tokens.shape)}")
        cached = layout_tensors or self._group_layout_tensors(layout, patch_tokens.device)
        member_indices = cached["member_indices"]
        member_mask = cached["member_mask"]
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
        return self._risk_head_scores(pooled)

    def _group_risk_scores_from_grouped(
        self,
        grouped_tokens: torch.Tensor,
        member_mask: torch.Tensor,
        member_counts: torch.Tensor,
        *,
        all_members_valid: bool = False,
    ) -> torch.Tensor:
        if self.group_risk_head is None:
            raise RuntimeError("risk_v1 requires a group risk head.")
        if all_members_valid:
            pooled = grouped_tokens.mean(dim=2)
        else:
            gathered = grouped_tokens * member_mask.view(1, member_mask.shape[0], member_mask.shape[1], 1).to(grouped_tokens.dtype)
            pooled = gathered.sum(dim=2) / member_counts.view(1, member_mask.shape[0], 1).to(grouped_tokens.dtype)
        return self._risk_head_scores(pooled)

    def _pooled_group_tokens_from_grouped(
        self,
        grouped_tokens: torch.Tensor,
        member_mask: torch.Tensor,
        member_counts: torch.Tensor,
        *,
        all_members_valid: bool = False,
    ) -> torch.Tensor:
        if all_members_valid:
            return grouped_tokens.mean(dim=2)
        gathered = grouped_tokens * member_mask.view(1, member_mask.shape[0], member_mask.shape[1], 1).to(grouped_tokens.dtype)
        return gathered.sum(dim=2) / member_counts.view(1, member_mask.shape[0], 1).to(grouped_tokens.dtype)

    def _patch_luma_tokens(
        self,
        frame_batch: torch.Tensor,
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        if frame_batch.shape[1] >= 3:
            luma = (
                0.299 * frame_batch[:, 0]
                + 0.587 * frame_batch[:, 1]
                + 0.114 * frame_batch[:, 2]
            )
        else:
            luma = frame_batch.mean(dim=1)
        pooled = F.avg_pool2d(
            luma.unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return pooled.reshape(frame_batch.shape[0], patch_height * patch_width)

    def _patch_dark_prior_map(
        self,
        frame_batch: torch.Tensor,
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        patch_luma = self._patch_luma_tokens(
            frame_batch,
            patch_height=patch_height,
            patch_width=patch_width,
        ).reshape(frame_batch.shape[0], 1, patch_height, patch_width)
        return (1.0 - patch_luma).clamp(0.0, 1.0)

    def _patch_luma_gradient_tokens(
        self,
        frame_batch: torch.Tensor,
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        if frame_batch.shape[1] >= 3:
            luma = (
                0.299 * frame_batch[:, 0]
                + 0.587 * frame_batch[:, 1]
                + 0.114 * frame_batch[:, 2]
            )
        else:
            luma = frame_batch.mean(dim=1)
        grad_x = F.pad(luma[:, :, 1:] - luma[:, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(luma[:, 1:, :] - luma[:, :-1, :], (0, 0, 0, 1))
        gradient = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
        pooled = F.avg_pool2d(
            gradient.unsqueeze(1),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return pooled.reshape(frame_batch.shape[0], patch_height * patch_width)

    def _group_temporal_stat(
        self,
        group_values: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        if group_values.ndim != 2:
            raise ValueError(f"Expected group_values [B*T, G], got {tuple(group_values.shape)}")
        reshaped = group_values.reshape(batch_size, num_frames, group_values.shape[1])
        temporal = reshaped.std(dim=1, unbiased=False)
        return temporal.reshape(batch_size, 1, group_values.shape[1]).expand(-1, num_frames, -1).reshape_as(group_values)

    def _group_feature_temporal_stat(
        self,
        pooled_tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        reshaped = pooled_tokens.reshape(batch_size, num_frames, pooled_tokens.shape[1], pooled_tokens.shape[2])
        temporal = reshaped.std(dim=1, unbiased=False).mean(dim=-1)
        return temporal.reshape(batch_size, 1, pooled_tokens.shape[1]).expand(-1, num_frames, -1).reshape(
            pooled_tokens.shape[0], pooled_tokens.shape[1]
        )

    def _compose_risk_v2_group_scores(
        self,
        grouped_tokens: torch.Tensor,
        grouped_coords: torch.Tensor,
        member_mask: torch.Tensor,
        member_counts: torch.Tensor,
        patch_tokens: torch.Tensor,
        layout: GroupLayout,
        frame_batch: torch.Tensor,
        patch_height: int,
        patch_width: int,
        batch_size: int,
        num_frames: int,
        collect_components: bool = True,
        all_members_valid: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        strategy = str(getattr(self.adaptive_config, "risk_merge_strategy", "low_risk")).lower()
        needs_cam_activation = strategy == "cam"
        needs_image_gradient = strategy in {"gradient", "grad"}
        collect_components = bool(collect_components or needs_cam_activation or needs_image_gradient)
        pooled_tokens = self._pooled_group_tokens_from_grouped(
            grouped_tokens,
            member_mask,
            member_counts,
            all_members_valid=all_members_valid,
        )
        components: dict[str, torch.Tensor] = {}
        final_scores: torch.Tensor | None = None

        simple_inference_risk = (
            not self.training
            and bool(getattr(self.adaptive_config, "risk_inference_fast_path", True))
            and strategy in {"low_risk", "ours", "high_risk"}
            and self.group_compressibility_head is None
            and self.group_task_preserve_head is None
            and self.task_token_shadow_head is None
        )
        if simple_inference_risk:
            if self.group_risk_head is not None:
                risk_prior = self._risk_head_scores(pooled_tokens)
                final_scores = float(self.adaptive_config.risk_feature_prior_scale) * risk_prior
                if collect_components:
                    components["risk_prior"] = risk_prior
                    components["final_group_scores"] = final_scores
                return final_scores, components
            final_scores = pooled_tokens.new_zeros((pooled_tokens.shape[0], pooled_tokens.shape[1]))
            if collect_components:
                components["final_group_scores"] = final_scores
            return final_scores, components

        if self.group_risk_head is not None:
            risk_prior = self._risk_head_scores(pooled_tokens)
            if collect_components:
                components["risk_prior"] = risk_prior
            final_scores = float(self.adaptive_config.risk_feature_prior_scale) * risk_prior
        if final_scores is None:
            final_scores = pooled_tokens.new_zeros((pooled_tokens.shape[0], pooled_tokens.shape[1]))

        if self.group_compressibility_head is not None:
            centered = grouped_tokens - pooled_tokens.unsqueeze(2)
            masked_sq = centered.pow(2) * member_mask.view(1, member_mask.shape[0], member_mask.shape[1], 1).to(grouped_tokens.dtype)
            feature_variance = masked_sq.mean(dim=(2, 3))
            compactness = torch.exp(-float(self.adaptive_config.compressibility_compactness_scale) * feature_variance)
            temporal_delta = self._group_feature_temporal_stat(pooled_tokens, batch_size=batch_size, num_frames=num_frames)
            temporal_stability = torch.exp(-float(self.adaptive_config.compressibility_temporal_scale) * temporal_delta)
            compressibility_stats = torch.stack((compactness, temporal_stability), dim=-1)
            compressibility = self.group_compressibility_head(pooled_tokens, compressibility_stats)
            if collect_components:
                components["compressibility"] = compressibility
                components["compressibility_compactness"] = compactness
                components["compressibility_temporal_stability"] = temporal_stability
            final_scores = final_scores - float(self.adaptive_config.compressibility_score_scale) * compressibility

        if self.group_task_preserve_head is not None and self.task_token_shadow_head is not None:
            token_shadow_logits = self.task_token_shadow_head(
                patch_tokens,
                patch_hw=(patch_height, patch_width),
                token_coords=layout.patch_coords.to(patch_tokens.device),
            )
            token_shadow_probs = torch.sigmoid(token_shadow_logits)
            group_mean_prob = compute_group_scores(token_shadow_probs, layout)
            if collect_components:
                components["cam_activation"] = group_mean_prob
            penumbra = (1.0 - torch.abs(2.0 * group_mean_prob - 1.0)).clamp(0.0, 1.0)
            member_indices = layout.group_member_indices.to(token_shadow_probs.device)
            member_mask_device = layout.group_member_mask.to(token_shadow_probs.device)
            gathered_probs = token_shadow_probs[:, member_indices.clamp_min(0)]
            gathered_probs = gathered_probs.masked_fill(~member_mask_device.unsqueeze(0), 0.0)
            max_prob = gathered_probs.amax(dim=2)
            min_prob = gathered_probs.masked_fill(~member_mask_device.unsqueeze(0), 1.0).amin(dim=2)
            boundary = (max_prob - min_prob).clamp_min(0.0)
            temporal_inconsistency = self._group_temporal_stat(group_mean_prob, batch_size=batch_size, num_frames=num_frames)
            patch_luma = self._patch_luma_tokens(frame_batch, patch_height=patch_height, patch_width=patch_width)
            group_luma = compute_group_scores(patch_luma, layout)
            dark_mask = (1.0 - (group_luma / max(float(self.adaptive_config.task_preserve_dark_threshold), 1e-6))).clamp(0.0, 1.0)
            dark_ambiguity = dark_mask * penumbra
            preserve_stats = torch.stack((boundary, penumbra, temporal_inconsistency, dark_ambiguity), dim=-1)
            learned_preserve = self.group_task_preserve_head(pooled_tokens, preserve_stats)
            heuristic_preserve = (
                float(self.adaptive_config.task_preserve_boundary_scale) * boundary
                + float(self.adaptive_config.task_preserve_penumbra_scale) * penumbra
                + float(self.adaptive_config.task_preserve_temporal_scale) * temporal_inconsistency
                + float(self.adaptive_config.task_preserve_dark_ambiguity_scale) * dark_ambiguity
            )
            task_preserve = learned_preserve + heuristic_preserve
            if collect_components:
                components["task_preserve"] = task_preserve
                components["task_boundary"] = boundary
                components["task_penumbra"] = penumbra
                components["task_temporal_inconsistency"] = temporal_inconsistency
                components["task_dark_ambiguity"] = dark_ambiguity
            final_scores = final_scores + float(self.adaptive_config.task_preserve_score_scale) * task_preserve

        if needs_cam_activation and "cam_activation" not in components:
            patch_luma = self._patch_luma_tokens(frame_batch, patch_height=patch_height, patch_width=patch_width)
            components["cam_activation"] = compute_group_scores((1.0 - patch_luma).clamp(0.0, 1.0), layout)
        if needs_image_gradient:
            patch_gradient = self._patch_luma_gradient_tokens(
                frame_batch,
                patch_height=patch_height,
                patch_width=patch_width,
            )
            components["image_gradient"] = compute_group_scores(patch_gradient, layout)

        if collect_components:
            components["final_group_scores"] = final_scores
        return final_scores, components

    def _ordered_group_indices_from_mask(self, merged_group_mask: torch.Tensor) -> torch.Tensor:
        num_frames_total, num_groups = merged_group_mask.shape
        base_order = torch.arange(num_groups, device=merged_group_mask.device, dtype=torch.long).view(1, num_groups)
        order_keys = merged_group_mask.logical_not().long() * num_groups + base_order
        return order_keys.argsort(dim=1)

    def _build_batch_merge_plan(
        self,
        merged_group_mask: torch.Tensor,
        member_indices: torch.Tensor,
        num_merge_groups: int | None = None,
    ) -> BatchMergePlan:
        num_frames_total, num_groups = merged_group_mask.shape
        group_width = int(member_indices.shape[1])
        if num_merge_groups is None:
            num_merge_groups = int(merged_group_mask[0].sum().item()) if merged_group_mask.numel() > 0 else 0
        if self.training and not torch.all(merged_group_mask.sum(dim=1) == num_merge_groups):
            raise ValueError("Batch merge requires identical merged-group counts across frames.")

        if num_merge_groups == 0:
            dst_index = self._runtime_arange(
                num_groups * group_width,
                merged_group_mask.device,
            ).unsqueeze(0).expand(num_frames_total, -1)
            return BatchMergePlan(
                dst_index=dst_index,
                old_to_new=dst_index,
                out_rows=num_groups * group_width,
                num_merge_groups=0,
            )

        if self._group_contract_extension_enabled and merged_group_mask.is_cuda:
            dst_index, old_to_new = group_contract_plan(
                merged_group_mask,
                member_indices,
                group_width,
                num_merge_groups,
            )
            return BatchMergePlan(
                dst_index=dst_index,
                old_to_new=old_to_new,
                out_rows=num_merge_groups + (num_groups - num_merge_groups) * group_width,
                num_merge_groups=num_merge_groups,
            )
        else:
            keep_group_mask = ~merged_group_mask
            merged_ranks = torch.cumsum(merged_group_mask.long(), dim=1) - 1
            kept_ranks = torch.cumsum(keep_group_mask.long(), dim=1) - 1
            member_offsets = self._runtime_arange(group_width, merged_group_mask.device).view(1, 1, group_width)
            dst_index = torch.where(
                merged_group_mask.unsqueeze(-1),
                merged_ranks.unsqueeze(-1),
                num_merge_groups + kept_ranks.unsqueeze(-1) * group_width + member_offsets,
            ).reshape(num_frames_total, num_groups * group_width)

        old_to_new = self._runtime_long_buffer(
            "old_to_new",
            (num_frames_total, num_groups * group_width),
            merged_group_mask.device,
        )
        scatter_indices = self._runtime_member_scatter_index(member_indices, num_frames_total)
        old_to_new.scatter_(1, scatter_indices, dst_index)
        return BatchMergePlan(
            dst_index=dst_index,
            old_to_new=old_to_new,
            out_rows=num_merge_groups + (num_groups - num_merge_groups) * group_width,
            num_merge_groups=num_merge_groups,
        )

    def _merge_grouped_batch_with_mask(
        self,
        grouped_tokens: torch.Tensor,
        grouped_coords: torch.Tensor,
        merged_group_mask: torch.Tensor,
        member_indices: torch.Tensor,
        grouped_sizes: torch.Tensor | None = None,
        num_merge_groups: int | None = None,
        return_token_sizes: bool = False,
        profile: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        num_frames_total, num_groups, group_width, channels = grouped_tokens.shape
        coord_channels = grouped_coords.shape[-1]
        if num_merge_groups is None:
            num_merge_groups = int(merged_group_mask[0].sum().item()) if merged_group_mask.numel() > 0 else 0
        if self.training and not torch.all(merged_group_mask.sum(dim=1) == num_merge_groups):
            raise ValueError("Batch merge requires identical merged-group counts across frames.")
        if num_merge_groups == 0:
            identity_mapping = self._runtime_arange(
                num_groups * group_width,
                grouped_tokens.device,
            ).unsqueeze(0).expand(num_frames_total, -1)
            return (
                grouped_tokens.reshape(num_frames_total, num_groups * group_width, channels),
                grouped_coords.reshape(num_frames_total, num_groups * group_width, coord_channels),
                identity_mapping,
                (
                    grouped_sizes.reshape(num_frames_total, num_groups * group_width)
                    if grouped_sizes is not None
                    else (
                        grouped_tokens.new_ones((num_frames_total, num_groups * group_width))
                        if return_token_sizes
                        else None
                    )
                ),
            )

        out_rows = num_merge_groups + (num_groups - num_merge_groups) * group_width
        can_use_fused_mask_contract = (
            self._group_contract_extension_enabled
            and bool(getattr(self.adaptive_config, "use_fused_mask_contract", False))
            and grouped_tokens.is_cuda
            and grouped_sizes is None
            and return_token_sizes
        )
        if can_use_fused_mask_contract:
            def _fused_contract_kernel():
                return group_contract_merge_mask_unweighted_with_sizes(
                    grouped_tokens,
                    grouped_coords,
                    merged_group_mask,
                    member_indices,
                    out_rows,
                    num_merge_groups,
                )

            if profile is not None:
                (
                    (new_patch_tokens, new_patch_coords, old_to_new, new_token_sizes),
                    contract_kernel_ms,
                ) = self._profile_section_ms(grouped_tokens.device, _fused_contract_kernel)
                profile["contract_kernel_ms"] += contract_kernel_ms
            else:
                new_patch_tokens, new_patch_coords, old_to_new, new_token_sizes = _fused_contract_kernel()
            return new_patch_tokens, new_patch_coords, old_to_new, new_token_sizes

        if profile is not None:
            merge_plan, merge_plan_ms = self._profile_section_ms(
                grouped_tokens.device,
                lambda: self._build_batch_merge_plan(
                    merged_group_mask,
                    member_indices,
                    num_merge_groups=num_merge_groups,
                ),
            )
            profile["merge_plan_ms"] += merge_plan_ms
        else:
            merge_plan = self._build_batch_merge_plan(merged_group_mask, member_indices, num_merge_groups=num_merge_groups)
        num_merge_groups = merge_plan.num_merge_groups
        dst_index = merge_plan.dst_index
        old_to_new = merge_plan.old_to_new
        out_rows = merge_plan.out_rows
        if self._group_contract_extension_enabled and grouped_tokens.is_cuda:
            def _contract_kernel():
                if grouped_sizes is not None:
                    contracted_sizes_mean = group_contract_merge(grouped_sizes.unsqueeze(-1), dst_index, out_rows).squeeze(-1)
                    weighted_grouped_tokens = grouped_tokens * grouped_sizes.unsqueeze(-1).to(grouped_tokens.dtype)
                    weighted_grouped_coords = grouped_coords * grouped_sizes.unsqueeze(-1).to(grouped_coords.dtype)
                    new_patch_tokens = group_contract_merge(weighted_grouped_tokens, dst_index, out_rows)
                    new_patch_coords = group_contract_merge(weighted_grouped_coords, dst_index, out_rows)
                    new_patch_tokens = new_patch_tokens / contracted_sizes_mean.unsqueeze(-1).clamp_min(1.0)
                    new_patch_coords = new_patch_coords / contracted_sizes_mean.unsqueeze(-1).clamp_min(1.0)
                    new_token_sizes = contracted_sizes_mean.clone()
                    new_token_sizes[:, :num_merge_groups] *= float(group_width)
                elif return_token_sizes:
                    new_patch_tokens, new_patch_coords, new_token_sizes = group_contract_merge_unweighted_with_sizes(
                        grouped_tokens,
                        grouped_coords,
                        dst_index,
                        out_rows,
                        num_merge_groups,
                    )
                else:
                    new_patch_tokens = group_contract_merge(grouped_tokens, dst_index, out_rows)
                    new_patch_coords = group_contract_merge(grouped_coords, dst_index, out_rows)
                    new_token_sizes = None
                return new_patch_tokens, new_patch_coords, new_token_sizes

            if profile is not None:
                (new_patch_tokens, new_patch_coords, new_token_sizes), contract_kernel_ms = self._profile_section_ms(
                    grouped_tokens.device,
                    _contract_kernel,
                )
                profile["contract_kernel_ms"] += contract_kernel_ms
            else:
                new_patch_tokens, new_patch_coords, new_token_sizes = _contract_kernel()
            return new_patch_tokens, new_patch_coords, old_to_new, new_token_sizes

        ordered_group_indices = self._ordered_group_indices_from_mask(merged_group_mask)
        token_order = ordered_group_indices.view(num_frames_total, num_groups, 1, 1).expand(-1, -1, group_width, channels)
        ordered_grouped_tokens = grouped_tokens.gather(1, token_order)
        coord_order = ordered_group_indices.view(num_frames_total, num_groups, 1, 1).expand(-1, -1, group_width, coord_channels)
        ordered_grouped_coords = grouped_coords.gather(1, coord_order)
        ordered_grouped_sizes = None
        if grouped_sizes is not None:
            size_order = ordered_group_indices.view(num_frames_total, num_groups, 1).expand(-1, -1, group_width)
            ordered_grouped_sizes = grouped_sizes.gather(1, size_order)

        if ordered_grouped_sizes is not None:
            merged_size_sums = ordered_grouped_sizes[:, :num_merge_groups].sum(dim=2).clamp_min(1.0)
            merged_tokens = (
                ordered_grouped_tokens[:, :num_merge_groups]
                * ordered_grouped_sizes[:, :num_merge_groups].unsqueeze(-1).to(ordered_grouped_tokens.dtype)
            ).sum(dim=2) / merged_size_sums.unsqueeze(-1)
            merged_coords = (
                ordered_grouped_coords[:, :num_merge_groups]
                * ordered_grouped_sizes[:, :num_merge_groups].unsqueeze(-1).to(ordered_grouped_coords.dtype)
            ).sum(dim=2) / merged_size_sums.unsqueeze(-1)
        else:
            merged_tokens = ordered_grouped_tokens[:, :num_merge_groups].mean(dim=2)
            merged_coords = ordered_grouped_coords[:, :num_merge_groups].mean(dim=2)
        kept_tokens = ordered_grouped_tokens[:, num_merge_groups:].reshape(
            num_frames_total,
            (num_groups - num_merge_groups) * group_width,
            channels,
        )
        kept_coords = ordered_grouped_coords[:, num_merge_groups:].reshape(
            num_frames_total,
            (num_groups - num_merge_groups) * group_width,
            coord_channels,
        )

        new_patch_tokens = torch.cat((merged_tokens, kept_tokens), dim=1)
        new_patch_coords = torch.cat((merged_coords, kept_coords), dim=1)
        new_token_sizes = None
        if ordered_grouped_sizes is not None:
            merged_sizes = ordered_grouped_sizes[:, :num_merge_groups].sum(dim=2)
            kept_sizes = ordered_grouped_sizes[:, num_merge_groups:].reshape(
                num_frames_total,
                (num_groups - num_merge_groups) * group_width,
            )
            new_token_sizes = torch.cat((merged_sizes, kept_sizes), dim=1)
        elif return_token_sizes:
            merged_sizes = grouped_tokens.new_full((num_frames_total, num_merge_groups), float(group_width))
            kept_sizes = grouped_tokens.new_ones((num_frames_total, (num_groups - num_merge_groups) * group_width))
            new_token_sizes = torch.cat((merged_sizes, kept_sizes), dim=1)

        return new_patch_tokens, new_patch_coords, old_to_new, new_token_sizes

    def _normalize_output(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_backbone_norm:
            return tokens
        if self.backbone.untie_cls_and_patch_norms:
            cls_reg = self.backbone.cls_norm(tokens[:, : self.token_prefix])
            patch = self.backbone.norm(tokens[:, self.token_prefix :])
            return torch.cat((cls_reg, patch), dim=1)
        return self.backbone.norm(tokens)

    def _tokens_to_feature_map(self, tokens: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
        patch_tokens = tokens[:, self.token_prefix :]
        batch_tokens = patch_tokens.shape[0]
        return patch_tokens.reshape(batch_tokens, patch_height, patch_width, -1).permute(0, 3, 1, 2).contiguous()

    def _frame_batch(self, clip: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int, int]:
        if clip.ndim != 5:
            raise ValueError(f"Expected [B, 3, T, H, W], got {tuple(clip.shape)}")
        batch_size, channels, num_frames, image_height, image_width = clip.shape
        if channels != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {channels}")
        frame_batch = clip.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, image_height, image_width)
        if not self.training and frame_batch.device.type == "cuda":
            frame_batch = frame_batch.contiguous(memory_format=torch.channels_last)
        return frame_batch, batch_size, channels, num_frames, image_height, image_width

    def _batched_rope_from_coords(
        self,
        coords: torch.Tensor,
        patch_height: int,
        patch_width: int,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.backbone.rope_embed
        periods = rope.periods
        coords = coords.to(device=periods.device, dtype=periods.dtype)

        if rope.normalize_coords == "max":
            max_hw = max(patch_height, patch_width)
            coords_h = coords[..., 0] / max_hw
            coords_w = coords[..., 1] / max_hw
        elif rope.normalize_coords == "min":
            min_hw = min(patch_height, patch_width)
            coords_h = coords[..., 0] / min_hw
            coords_w = coords[..., 1] / min_hw
        elif rope.normalize_coords == "separate":
            coords_h = coords[..., 0] / patch_height
            coords_w = coords[..., 1] / patch_width
        else:
            raise ValueError(f"Unsupported rope coordinate normalization: {rope.normalize_coords}")

        normalized = torch.stack((coords_h, coords_w), dim=-1)
        normalized = 2.0 * normalized - 1.0
        angles = 2.0 * math.pi * normalized[..., None] / periods[None, None, None, :]
        angles = angles.flatten(2, 3).tile((1, 1, 2))
        sin = torch.sin(angles).unsqueeze(1)
        cos = torch.cos(angles).unsqueeze(1)
        if dtype is not None:
            sin = sin.to(dtype=dtype)
            cos = cos.to(dtype=dtype)
        return sin, cos

    def _runtime_rope_dtype(self, tokens: torch.Tensor) -> torch.dtype:
        if self.training or tokens.device.type != "cuda":
            return self.backbone.rope_embed.periods.dtype
        try:
            autocast_enabled = torch.is_autocast_enabled("cuda")
        except TypeError:
            autocast_enabled = torch.is_autocast_enabled()
        if autocast_enabled:
            if hasattr(torch, "get_autocast_dtype"):
                return torch.get_autocast_dtype("cuda")
            return torch.get_autocast_gpu_dtype()
        return tokens.dtype

    def _restore_tokens_batch(self, tokens: torch.Tensor, original_to_current: torch.Tensor) -> torch.Tensor:
        if original_to_current is None:
            return tokens
        restored_patch = self._restore_patch_tokens_batch(tokens[:, self.token_prefix :], original_to_current)
        return torch.cat((tokens[:, : self.token_prefix], restored_patch), dim=1)

    def _restore_patch_tokens_batch(
        self,
        patch_tokens: torch.Tensor,
        original_to_current: torch.Tensor | None,
        batch_restore_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if original_to_current is None:
            return patch_tokens
        del batch_restore_index
        if self._group_contract_extension_enabled and patch_tokens.is_cuda:
            return group_restore(patch_tokens, original_to_current)
        gather_index = original_to_current.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        return patch_tokens.gather(1, gather_index)

    def _restore_patch_tokens_to_feature_map(
        self,
        patch_tokens: torch.Tensor,
        original_to_current: torch.Tensor | None,
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        if original_to_current is None:
            patch_maps = patch_tokens.reshape(patch_tokens.shape[0], patch_height, patch_width, -1)
            return patch_maps.permute(0, 3, 1, 2).contiguous()
        if self._group_contract_extension_enabled and patch_tokens.is_cuda:
            return group_restore_to_map(
                patch_tokens,
                original_to_current,
                patch_height,
                patch_width,
            )
        restored_patch_tokens = self._restore_patch_tokens_batch(patch_tokens, original_to_current)
        patch_maps = restored_patch_tokens.reshape(restored_patch_tokens.shape[0], patch_height, patch_width, -1)
        return patch_maps.permute(0, 3, 1, 2).contiguous()

    def _collect_batch_output(
        self,
        tokens: torch.Tensor,
        original_to_current: torch.Tensor | None,
        patch_height: int,
        patch_width: int,
        batch_size: int,
        num_frames: int,
        batch_restore_index: torch.Tensor | None = None,
        collect_cls: bool = False,
        collect_profile: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del batch_restore_index
        if collect_profile is not None:
            device = tokens.device
            normalized, normalize_ms = self._profile_section_ms(device, lambda: self._normalize_output(tokens))
            collect_profile["collect_normalize_ms"] += normalize_ms
            patch_maps, restore_to_map_ms = self._profile_section_ms(
                device,
                lambda: self._restore_patch_tokens_to_feature_map(
                    normalized[:, self.token_prefix :],
                    original_to_current,
                    patch_height,
                    patch_width,
                ),
            )
            collect_profile["restore_to_map_ms"] += restore_to_map_ms
        else:
            normalized = self._normalize_output(tokens)
            patch_maps = self._restore_patch_tokens_to_feature_map(
                normalized[:, self.token_prefix :],
                original_to_current,
                patch_height,
                patch_width,
            )
        feature_map = patch_maps.reshape(batch_size, num_frames, self.embed_dim, patch_height, patch_width)
        cls_tokens = None
        if collect_cls:
            if collect_profile is not None:
                cls_tokens, collect_cls_ms = self._profile_section_ms(
                    tokens.device,
                    lambda: normalized[:, 0].reshape(batch_size, num_frames, self.embed_dim),
                )
                collect_profile["collect_cls_ms"] += collect_cls_ms
            else:
                cls_tokens = normalized[:, 0].reshape(batch_size, num_frames, self.embed_dim)
        return feature_map, cls_tokens

    def _run_block_batch(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        token_sizes: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
        size_bias: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        tokens_for_block = tokens
        if token_sizes is not None and self._size_bias_enabled():
            patch_scale, patch_bias = (
                size_bias if size_bias is not None else self._build_size_bias(token_sizes.to(tokens.dtype))
            )
            tokens_for_block = torch.cat(
                (
                    tokens[:, : self.token_prefix],
                    tokens[:, self.token_prefix :] * patch_scale + patch_bias,
                ),
                dim=1,
            )
        backbone_context = nullcontext if self.has_trainable_backbone else torch.no_grad
        with backbone_context():
            return block(tokens_for_block, rope, attn_bias)

    def _run_compact_block_batch(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        token_sizes: torch.Tensor | None,
        attn_bias: torch.Tensor | None,
        size_bias: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        # Hot sparse-tail path: all spatial tokens are already compact, so avoid
        # dense-tail checks here and reuse precomputed size/attention metadata.
        if size_bias is not None:
            patch_scale, patch_bias = size_bias
            tokens_for_block = torch.cat(
                (
                    tokens[:, : self.token_prefix],
                    tokens[:, self.token_prefix :] * patch_scale + patch_bias,
                ),
                dim=1,
            )
        elif token_sizes is not None and self._size_bias_enabled():
            patch_scale, patch_bias = self._build_size_bias(token_sizes.to(tokens.dtype))
            tokens_for_block = torch.cat(
                (
                    tokens[:, : self.token_prefix],
                    tokens[:, self.token_prefix :] * patch_scale + patch_bias,
                ),
                dim=1,
            )
        else:
            tokens_for_block = tokens
        backbone_context = nullcontext if self.has_trainable_backbone else torch.no_grad
        with backbone_context():
            return block(tokens_for_block, rope, attn_bias)

    def _prepare_compact_block_input(
        self,
        tokens: torch.Tensor,
        token_sizes: torch.Tensor | None,
        size_bias: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if size_bias is not None:
            patch_scale, patch_bias = size_bias
            return torch.cat(
                (
                    tokens[:, : self.token_prefix],
                    tokens[:, self.token_prefix :] * patch_scale + patch_bias,
                ),
                dim=1,
            )
        if token_sizes is not None and self._size_bias_enabled():
            patch_scale, patch_bias = self._build_size_bias(token_sizes.to(tokens.dtype))
            return torch.cat(
                (
                    tokens[:, : self.token_prefix],
                    tokens[:, self.token_prefix :] * patch_scale + patch_bias,
                ),
                dim=1,
            )
        return tokens

    def _run_profiled_compact_block_batch(
        self,
        block: nn.Module,
        tokens: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        token_sizes: torch.Tensor | None,
        attn_bias: torch.Tensor | None,
        size_bias: tuple[torch.Tensor, torch.Tensor] | None,
        profile: dict[str, float],
    ) -> torch.Tensor:
        if self.training and getattr(block, "sample_drop_ratio", 0.0) > 0.0:
            return self._run_compact_block_batch(
                block,
                tokens,
                rope,
                token_sizes=token_sizes,
                attn_bias=attn_bias,
                size_bias=size_bias,
            )
        if not all(hasattr(block, attr) for attr in ("norm1", "attn", "ls1", "norm2", "mlp", "ls2")):
            return self._run_compact_block_batch(
                block,
                tokens,
                rope,
                token_sizes=token_sizes,
                attn_bias=attn_bias,
                size_bias=size_bias,
            )
        if not all(hasattr(block.attn, attr) for attr in ("qkv", "proj", "proj_drop", "num_heads", "apply_rope")):
            return self._run_compact_block_batch(
                block,
                tokens,
                rope,
                token_sizes=token_sizes,
                attn_bias=attn_bias,
                size_bias=size_bias,
            )

        device = tokens.device
        cuda_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

        def timed(key: str, fn):
            if not self.enable_speed_profile:
                return fn()
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream(device))
                result = fn()
                end_event.record(torch.cuda.current_stream(device))
                cuda_events.append((key, start_event, end_event))
                return result
            start_time = perf_counter()
            result = fn()
            profile[key] += float((perf_counter() - start_time) * 1000.0)
            return result

        tokens_for_block = timed(
            "post_size_bias_apply_ms",
            lambda: self._prepare_compact_block_input(tokens, token_sizes, size_bias),
        )
        norm1 = timed("post_norm1_ms", lambda: block.norm1(tokens_for_block))
        qkv = timed("post_qkv_ms", lambda: block.attn.qkv(norm1))

        def _qkv_layout():
            batch_size, num_tokens, _ = qkv.shape
            channels = block.attn.qkv.in_features
            qkv_view = qkv.reshape(batch_size, num_tokens, 3, block.attn.num_heads, channels // block.attn.num_heads)
            q, k, v = torch.unbind(qkv_view, 2)
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), batch_size, num_tokens, channels

        q, k, v, batch_size, num_tokens, channels = timed("post_layout_ms", _qkv_layout)
        if rope is not None:
            q, k = timed("post_rope_apply_ms", lambda: block.attn.apply_rope(q, k, rope))
        attn_out = timed(
            "post_attn_ms",
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias),
        )

        attn_values = timed(
            "post_layout_ms",
            lambda: attn_out.transpose(1, 2).reshape(batch_size, num_tokens, channels),
        )
        attn_projected = timed(
            "post_attn_proj_ms",
            lambda: block.attn.proj_drop(block.attn.proj(attn_values)),
        )
        x_attn = timed(
            "post_residual_ms",
            lambda: tokens_for_block + block.ls1(attn_projected),
        )

        norm2 = timed("post_norm2_ms", lambda: block.norm2(x_attn))

        if all(hasattr(block.mlp, attr) for attr in ("fc1", "act", "fc2", "drop")):
            mlp_hidden = timed("post_mlp_fc1_ms", lambda: block.mlp.fc1(norm2))
            mlp_hidden = timed("post_mlp_act_ms", lambda: block.mlp.drop(block.mlp.act(mlp_hidden)))
            mlp_out = timed("post_mlp_fc2_ms", lambda: block.mlp.drop(block.mlp.fc2(mlp_hidden)))
        else:
            mlp_out = timed("post_mlp_fc1_ms", lambda: block.mlp(norm2))

        x_ffn = timed("post_residual_ms", lambda: x_attn + block.ls2(mlp_out))
        if cuda_events:
            cuda_events[-1][2].synchronize()
            for key, start_event, end_event in cuda_events:
                profile[key] += float(start_event.elapsed_time(end_event))
        return x_ffn

    def _risk_hard_gate_mask_batch(
        self,
        group_scores: torch.Tensor,
        layout: GroupLayout,
        protect_ratio: float | None = None,
    ) -> torch.Tensor:
        num_frames_total, num_groups = group_scores.shape
        protected_mask = torch.zeros(num_frames_total, num_groups, dtype=torch.bool, device=group_scores.device)
        protect_ratio = float(self.adaptive_config.risk_hard_gate_ratio if protect_ratio is None else protect_ratio)
        if protect_ratio <= 0.0:
            return protected_mask
        num_protected = int(math.ceil(protect_ratio * layout.num_groups))
        num_protected = max(0, min(num_protected, layout.num_groups))
        if num_protected == 0:
            return protected_mask
        protected_indices = torch.topk(group_scores, k=num_protected, dim=1, largest=True).indices
        protected_mask.scatter_(1, protected_indices, True)
        return protected_mask

    def _candidate_group_mask_batch_with_protection(
        self,
        group_scores: torch.Tensor,
        keep_ratio: float,
        layout: GroupLayout,
        protected_mask: torch.Tensor | None,
        group_score_components: dict[str, torch.Tensor] | None = None,
        num_merge_groups: int | None = None,
    ) -> torch.Tensor:
        num_frames_total, num_groups = group_scores.shape
        if num_merge_groups is None:
            num_merge_groups = int((1.0 - keep_ratio) * num_groups)
            num_merge_groups = max(0, min(num_merge_groups, max(num_groups - 1, 0)))
        merged_mask = torch.zeros(num_frames_total, num_groups, dtype=torch.bool, device=group_scores.device)
        if num_merge_groups == 0:
            return merged_mask
        has_protection = protected_mask is not None
        if has_protection:
            eligible_groups = int((~protected_mask).sum(dim=1).amin().item()) if protected_mask.numel() > 0 else num_groups
            num_merge_groups = min(num_merge_groups, eligible_groups)
            if num_merge_groups == 0:
                return merged_mask

        strategy = str(getattr(self.adaptive_config, "risk_merge_strategy", "low_risk")).lower()
        components = group_score_components or {}
        if strategy in {"low_risk", "ours"}:
            selection_scores = group_scores.masked_fill(protected_mask, float("inf")) if has_protection else group_scores
            merge_indices = torch.topk(selection_scores, k=num_merge_groups, dim=1, largest=False).indices
        elif strategy == "high_risk":
            selection_scores = group_scores.masked_fill(protected_mask, float("-inf")) if has_protection else group_scores
            merge_indices = torch.topk(selection_scores, k=num_merge_groups, dim=1, largest=True).indices
        elif strategy == "random":
            random_scores = torch.rand_like(group_scores)
            selection_scores = random_scores.masked_fill(protected_mask, float("inf")) if has_protection else random_scores
            merge_indices = torch.topk(selection_scores, k=num_merge_groups, dim=1, largest=False).indices
        elif strategy == "cam":
            cam_scores = components.get("cam_activation", group_scores)
            selection_scores = cam_scores.masked_fill(protected_mask, float("inf")) if has_protection else cam_scores
            merge_indices = torch.topk(selection_scores, k=num_merge_groups, dim=1, largest=False).indices
        elif strategy in {"gradient", "grad"}:
            gradient_scores = components.get("image_gradient", group_scores)
            selection_scores = gradient_scores.masked_fill(protected_mask, float("inf")) if has_protection else gradient_scores
            merge_indices = torch.topk(selection_scores, k=num_merge_groups, dim=1, largest=False).indices
        else:
            raise ValueError(
                f"Unsupported risk_merge_strategy={strategy!r}. "
                "Expected one of: low_risk, ours, random, high_risk, cam, gradient."
            )
        merged_mask.scatter_(1, merge_indices, True)
        if has_protection:
            merged_mask &= ~protected_mask
        return merged_mask

    def _simple_merge_single(
        self,
        patch_tokens: torch.Tensor,
        patch_coords: torch.Tensor,
        group_scores: torch.Tensor,
        layout: GroupLayout,
        keep_ratio: float,
        layout_tensors: dict[str, torch.Tensor] | None = None,
        token_sizes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        merged_group_mask = self._candidate_group_mask(group_scores, keep_ratio=keep_ratio, layout=layout)
        cached = layout_tensors or self._group_layout_tensors(layout, patch_tokens.device)
        member_indices = cached["member_indices"]
        member_mask = cached["member_mask"]
        layout_coords = cached["patch_coords"]
        group_centers = cached["group_centers"]

        new_tokens = []
        new_coords = []
        new_sizes = [] if token_sizes is not None else None
        old_to_new = torch.empty(patch_tokens.shape[0], dtype=torch.long, device=patch_tokens.device)
        next_index = 0

        for group_index in range(layout.num_groups):
            members = member_indices[group_index][member_mask[group_index]]
            if bool(merged_group_mask[group_index].item()):
                if token_sizes is not None:
                    group_member_sizes = token_sizes[members]
                    total_size = group_member_sizes.sum().clamp_min(1.0)
                    weights = (group_member_sizes / total_size).to(patch_tokens.dtype)
                    merged_token = (patch_tokens[members] * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
                    merged_coord = (layout_coords[members] * weights.unsqueeze(-1).to(layout_coords.dtype)).sum(dim=0, keepdim=True)
                    merged_size = total_size.view(1)
                else:
                    merged_token = patch_tokens[members].mean(dim=0, keepdim=True)
                    merged_coord = group_centers[group_index : group_index + 1]
                    merged_size = None
                new_tokens.append(merged_token)
                new_coords.append(merged_coord)
                if new_sizes is not None and merged_size is not None:
                    new_sizes.append(merged_size)
                old_to_new[members] = next_index
                next_index += 1
            else:
                kept_tokens = patch_tokens[members]
                new_tokens.append(kept_tokens)
                new_coords.append(layout_coords[members])
                if new_sizes is not None:
                    new_sizes.append(token_sizes[members])
                old_to_new[members] = torch.arange(
                    next_index,
                    next_index + members.numel(),
                    device=patch_tokens.device,
                    dtype=torch.long,
                )
                next_index += members.numel()

        return (
            torch.cat(new_tokens, dim=0),
            torch.cat(new_coords, dim=0),
            old_to_new,
            merged_group_mask,
            torch.cat(new_sizes, dim=0) if new_sizes is not None else None,
        )

    def _simple_merge_batch_uniform(
        self,
        patch_tokens: torch.Tensor,
        group_scores: torch.Tensor,
        layout: GroupLayout,
        keep_ratio: float,
        layout_tensors: dict[str, torch.Tensor] | None = None,
        token_sizes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cached = layout_tensors or self._group_layout_tensors(layout, patch_tokens.device)
        member_indices = cached["member_indices"]
        member_mask = cached["member_mask"]
        group_sizes = member_mask.sum(dim=1)
        if not torch.all(group_sizes == group_sizes[0]):
            raise ValueError("Uniform batch merge requires all groups to have the same size.")

        num_frames_total, _, channels = patch_tokens.shape
        num_groups, group_width = member_indices.shape
        num_merge_groups = int((1.0 - keep_ratio) * layout.num_groups)
        num_merge_groups = max(0, min(num_merge_groups, max(layout.num_groups - 1, 0)))
        if num_merge_groups == 0:
            identity_mapping = self._runtime_arange(patch_tokens.shape[1], patch_tokens.device)
            return (
                patch_tokens,
                cached["patch_coords"].unsqueeze(0).expand(num_frames_total, -1, -1),
                identity_mapping.unsqueeze(0).expand(num_frames_total, -1),
                torch.zeros(num_frames_total, num_groups, dtype=torch.bool, device=patch_tokens.device),
                token_sizes,
            )

        gather_index = member_indices.view(1, num_groups, group_width, 1).expand(num_frames_total, -1, -1, channels)
        grouped_tokens = patch_tokens.gather(1, gather_index.reshape(num_frames_total, num_groups * group_width, channels))
        grouped_tokens = grouped_tokens.reshape(num_frames_total, num_groups, group_width, channels)

        patch_coords = cached["patch_coords"]
        grouped_coords = patch_coords[member_indices.clamp_min(0)].unsqueeze(0).expand(num_frames_total, -1, -1, -1)
        grouped_sizes = None
        if token_sizes is not None:
            grouped_sizes = token_sizes.gather(
                1,
                member_indices.view(1, num_groups * group_width).expand(num_frames_total, -1),
            ).reshape(num_frames_total, num_groups, group_width)

        merge_group_indices = torch.topk(group_scores, k=num_merge_groups, dim=1, largest=False).indices
        merged_group_mask = torch.zeros(num_frames_total, num_groups, dtype=torch.bool, device=patch_tokens.device)
        merged_group_mask.scatter_(1, merge_group_indices, True)
        new_patch_tokens, new_patch_coords, old_to_new, new_token_sizes = self._merge_grouped_batch_with_mask(
            grouped_tokens=grouped_tokens,
            grouped_coords=grouped_coords,
            merged_group_mask=merged_group_mask,
            member_indices=member_indices,
            grouped_sizes=grouped_sizes,
        )

        return new_patch_tokens, new_patch_coords, old_to_new, merged_group_mask, new_token_sizes

    def _simple_merge_batch_with_mask(
        self,
        patch_tokens: torch.Tensor,
        merged_group_mask: torch.Tensor,
        layout: GroupLayout,
        layout_tensors: dict[str, torch.Tensor] | None = None,
        grouped_tokens: torch.Tensor | None = None,
        grouped_coords: torch.Tensor | None = None,
        member_mask: torch.Tensor | None = None,
        token_sizes: torch.Tensor | None = None,
        grouped_sizes: torch.Tensor | None = None,
        num_merge_groups: int | None = None,
        return_token_sizes: bool = False,
        profile: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        cached = layout_tensors or self._group_layout_tensors(layout, patch_tokens.device)
        member_indices = cached["member_indices"]
        member_mask = member_mask if member_mask is not None else cached["member_mask"]
        group_sizes = member_mask.sum(dim=1)
        if self.training and not torch.all(group_sizes == group_sizes[0]):
            raise ValueError("Batch merge requires all groups to have the same size.")

        num_frames_total, _, channels = patch_tokens.shape
        num_groups, group_width = member_indices.shape
        if num_merge_groups is None:
            num_merge_groups = int(merged_group_mask[0].sum().item()) if merged_group_mask.numel() > 0 else 0
        if self.training and not torch.all(merged_group_mask.sum(dim=1) == num_merge_groups):
            raise ValueError("Batch merge requires identical merged-group counts across frames.")
        if num_merge_groups == 0:
            identity_mapping = self._runtime_arange(patch_tokens.shape[1], patch_tokens.device)
            return (
                patch_tokens,
                cached["patch_coords"].unsqueeze(0).expand(num_frames_total, -1, -1),
                identity_mapping.unsqueeze(0).expand(num_frames_total, -1),
                token_sizes if token_sizes is not None else (
                    patch_tokens.new_ones((num_frames_total, patch_tokens.shape[1]))
                    if return_token_sizes
                    else None
                ),
            )

        if grouped_tokens is None or grouped_coords is None:
            gather_index = cached["member_indices_clamped"].view(1, num_groups, group_width, 1).expand(num_frames_total, -1, -1, channels)
            grouped_tokens = patch_tokens.gather(1, gather_index.reshape(num_frames_total, num_groups * group_width, channels))
            grouped_tokens = grouped_tokens.reshape(num_frames_total, num_groups, group_width, channels)
            grouped_coords = cached["patch_coords_by_group"].unsqueeze(0).expand(num_frames_total, -1, -1, -1)
        if grouped_sizes is None and token_sizes is not None:
            grouped_sizes = token_sizes.gather(
                1,
                member_indices.view(1, num_groups * group_width).expand(num_frames_total, -1),
            ).reshape(num_frames_total, num_groups, group_width)
        return self._merge_grouped_batch_with_mask(
            grouped_tokens=grouped_tokens,
            grouped_coords=grouped_coords,
            merged_group_mask=merged_group_mask,
            member_indices=member_indices,
            grouped_sizes=grouped_sizes,
            num_merge_groups=num_merge_groups,
            return_token_sizes=return_token_sizes,
            profile=profile,
        )

    def _forward_baseline(self, clip: torch.Tensor) -> dict[str, object]:
        if self.decoder_variant == "slim_dpt":
            frame_batch, batch_size, _, num_frames, _, _ = self._frame_batch(clip)
            baseline_start = self._profile_marker(frame_batch.device)
            context = nullcontext if self.has_trainable_backbone else torch.no_grad
            with context():
                intermediate = self.backbone.get_intermediate_layers(
                    frame_batch,
                    n=self.out_indices,
                    reshape=True,
                    return_class_token=True,
                    norm=self.use_backbone_norm,
                )
            baseline_end = self._profile_marker(frame_batch.device)

            feature_maps: list[torch.Tensor] = []
            cls_tokens: list[torch.Tensor] = []
            for patch_map, cls_token in intermediate:
                _, feat_channels, feat_height, feat_width = patch_map.shape
                feature_maps.append(
                    patch_map.reshape(batch_size, num_frames, feat_channels, feat_height, feat_width)
                )
                cls_tokens.append(cls_token.reshape(batch_size, num_frames, cls_token.shape[-1]))

            if self.enable_speed_profile:
                self.last_speed_profile = {
                    "pre_merge_ms": self._profile_elapsed_ms(baseline_start, baseline_end, frame_batch.device),
                    "merge_ms": 0.0,
                    "post_merge_ms": 0.0,
                }

            return {
                "feature_maps": feature_maps,
                "cls_tokens": cls_tokens,
                "sparse_decoder_inputs": None,
                "adaptive_outputs": None,
                "existence_outputs": self._collect_existence_outputs(cls_tokens) if cls_tokens else None,
            }
        return self._forward_baseline_sparse_decoder(clip)

    def _forward_baseline_sparse_decoder(self, clip: torch.Tensor) -> dict[str, object]:
        frame_batch, batch_size, _, num_frames, _, _ = self._frame_batch(clip)
        device = frame_batch.device
        baseline_start = self._profile_marker(device)
        with (nullcontext() if self.has_trainable_backbone else torch.no_grad()):
            tokens, (patch_height, patch_width) = self.backbone.prepare_tokens_with_masks(frame_batch)

        base_rope = self._base_rope(patch_height, patch_width, tokens.device)
        layout = self._group_layout(patch_height, patch_width)
        layout_tensors = self._group_layout_tensors(layout, tokens.device)
        output_feature_maps: dict[int, torch.Tensor] = {}
        output_cls_tokens: dict[int, torch.Tensor] = {}
        collect_cls_outputs = self.existence_head is not None

        for block_index, block in enumerate(self.backbone.blocks):
            tokens = self._run_block_batch(block, tokens, base_rope)
            should_collect_feature_map = (
                block_index in self.out_indices
                and self._decoder_requires_dense_sources()
            )
            if should_collect_feature_map:
                feature_map, cls_tokens = self._collect_batch_output(
                    tokens=tokens,
                    original_to_current=None,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    collect_cls=collect_cls_outputs,
                )
                output_feature_maps[block_index] = feature_map
                if collect_cls_outputs and cls_tokens is not None:
                    output_cls_tokens[block_index] = cls_tokens
            elif collect_cls_outputs and block_index in self.out_indices:
                normalized = self._normalize_output(tokens)
                output_cls_tokens[block_index] = normalized[:, 0].reshape(batch_size, num_frames, self.embed_dim)

        sparse_decoder_inputs = self._build_sparse_decoder_inputs(
            output_feature_maps=output_feature_maps,
            merge_index=self.n_blocks - 1,
            tokens=tokens,
            patch_height=patch_height,
            patch_width=patch_width,
            layout_tensors=layout_tensors,
            patch_coords_batch=None,
            token_sizes_batch=None,
            original_to_current=None,
            num_dense_sources=(int(getattr(self, "decoder_simple_num_dense_sources", 2)) if self.decoder_variant == "decoder_simple" else 0),
            frame_batch=frame_batch,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        cls_tokens = [output_cls_tokens[out_index] for out_index in sorted(output_cls_tokens)] if collect_cls_outputs else []
        baseline_end = self._profile_marker(device)
        if self.enable_speed_profile:
            self.last_speed_profile = {
                "pre_merge_ms": self._profile_elapsed_ms(baseline_start, baseline_end, device),
                "merge_ms": 0.0,
                "post_merge_ms": 0.0,
            }
        return {
            "feature_maps": [],
            "cls_tokens": cls_tokens,
            "sparse_decoder_inputs": sparse_decoder_inputs,
            "adaptive_outputs": {
                "enabled": False,
                "runtime_mode": "baseline_sparse_decoder",
                "merge_block_indices": tuple(),
                "keep_ratios_summary": tuple(),
                "final_token_keep_ratio": 1.0,
                "final_token_saved_ratio": 0.0,
                "final_interface_token_keep_ratio": 1.0,
                "merge_semantics": "none",
                "execution_granularity": "batch",
                "token_transform_reducer_enabled": bool(self.token_transform_reducer is not None),
                "temporal_consensus_correction_enabled": bool(self.temporal_consensus_correction is not None),
            },
            "existence_outputs": self._collect_existence_outputs(cls_tokens) if collect_cls_outputs else None,
        }

    def _rope_from_coords(self, coords: torch.Tensor, patch_height: int, patch_width: int) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.backbone.rope_embed
        periods = rope.periods
        coords = coords.to(device=periods.device, dtype=periods.dtype)

        if rope.normalize_coords == "max":
            max_hw = max(patch_height, patch_width)
            coords_h = coords[:, 0] / max_hw
            coords_w = coords[:, 1] / max_hw
        elif rope.normalize_coords == "min":
            min_hw = min(patch_height, patch_width)
            coords_h = coords[:, 0] / min_hw
            coords_w = coords[:, 1] / min_hw
        elif rope.normalize_coords == "separate":
            coords_h = coords[:, 0] / patch_height
            coords_w = coords[:, 1] / patch_width
        else:
            raise ValueError(f"Unsupported rope coordinate normalization: {rope.normalize_coords}")

        normalized = torch.stack((coords_h, coords_w), dim=-1)
        normalized = 2.0 * normalized - 1.0
        angles = 2.0 * math.pi * normalized[:, :, None] / periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)

    def _resolve_merge_schedule(self) -> tuple[int, ...]:
        merge_indices = self.adaptive_config.merge_block_indices or (self.adaptive_config.merge_block_index,)
        merge_indices = tuple(sorted(set(int(index) for index in merge_indices)))
        for merge_index in merge_indices:
            if not (0 <= merge_index < self.n_blocks):
                raise ValueError(f"merge block index must be within [0, {self.n_blocks - 1}], got {merge_index}")
        return merge_indices

    def _resolve_tail_stop_block_index(self, merge_index: int) -> int:
        tail_stop_block_index = int(getattr(self.adaptive_config, "tail_stop_block_index", -1))
        if tail_stop_block_index < 0:
            return self.n_blocks - 1
        if not self._uses_sparse_decoder():
            raise ValueError("tail_stop_block_index is only supported with sparse decoder variants.")
        if not (0 <= tail_stop_block_index < self.n_blocks):
            raise ValueError(
                f"tail_stop_block_index must be within [0, {self.n_blocks - 1}], got {tail_stop_block_index}"
            )
        if tail_stop_block_index < merge_index:
            raise ValueError(
                f"tail_stop_block_index must be >= merge_block_index for sparse decoder, got "
                f"{tail_stop_block_index} < {merge_index}"
            )
        return tail_stop_block_index

    def _resolve_risk_v1_schedule(self) -> tuple[tuple[int, ...], int]:
        persistent_merge_index = int(self.adaptive_config.merge_block_index)
        if not (0 <= persistent_merge_index < self.n_blocks):
            raise ValueError(
                f"merge block index must be within [0, {self.n_blocks - 1}], got {persistent_merge_index}"
            )
        transient_merge_indices = tuple(
            sorted(
                set(
                    int(index)
                    for index in (self.adaptive_config.merge_block_indices or ())
                    if int(index) != persistent_merge_index
                )
            )
        )
        for merge_index in transient_merge_indices:
            if not (0 <= merge_index < self.n_blocks):
                raise ValueError(
                    f"merge block index must be within [0, {self.n_blocks - 1}], got {merge_index}"
                )
        return transient_merge_indices, persistent_merge_index

    def _should_hard_bypass_risk_v1(self) -> bool:
        if not bool(getattr(self.adaptive_config, "risk_keep_one_bypass_enabled", True)):
            return False
        if bool(getattr(self.adaptive_config, "risk_dynamic_keep_enabled", False)):
            return False
        keep_ratios = tuple(float(value) for value in (getattr(self.adaptive_config, "fixed_keep_ratios", ()) or ()))
        if not keep_ratios:
            keep_ratios = (float(getattr(self.adaptive_config, "fixed_keep_ratio", 1.0)),)
        threshold = float(getattr(self.adaptive_config, "risk_keep_one_bypass_threshold", 0.999))
        return all(keep_ratio >= threshold for keep_ratio in keep_ratios)

    def _should_commit_persistent_sparse_tail(self, keep_ratio: float) -> bool:
        commit_threshold = float(getattr(self.adaptive_config, "risk_persistent_commit_keep_threshold", 0.90))
        return keep_ratio < commit_threshold

    def _fixed_group_keep_ratio_from_value(self, keep_ratio: float, layout: GroupLayout) -> float:
        keep_ratio = min(max(keep_ratio, 0.0), 1.0)
        if not bool(getattr(self.adaptive_config, "fixed_keep_ratio_is_token_keep", False)):
            return keep_ratio
        group_width = max(int(layout.max_group_size), 1)
        if group_width <= 1:
            return keep_ratio
        # A merged group keeps one representative token, while an unmerged group keeps
        # all group_width tokens: true_keep = (1 + (group_width - 1) * group_keep) / group_width.
        group_keep = (group_width * keep_ratio - 1.0) / float(group_width - 1)
        return min(max(group_keep, 0.0), 1.0)

    def _fixed_group_keep_ratio(self, layout: GroupLayout) -> float:
        keep_ratio = float(getattr(self.adaptive_config, "fixed_keep_ratio", 1.0))
        return self._fixed_group_keep_ratio_from_value(keep_ratio, layout)

    def _fixed_group_keep_ratios_by_stage(
        self,
        layout: GroupLayout,
        merge_indices: tuple[int, ...],
    ) -> dict[int, float]:
        raw_keep_ratios = tuple(
            float(value)
            for value in (getattr(self.adaptive_config, "fixed_keep_ratios", ()) or ())
        )
        if raw_keep_ratios and len(raw_keep_ratios) != len(merge_indices):
            raise ValueError(
                "fixed_keep_ratios must be empty or have the same length as merge stages: "
                f"got {len(raw_keep_ratios)} ratios for {len(merge_indices)} stages "
                f"(merge_indices={merge_indices})"
            )
        keep_ratios = raw_keep_ratios or tuple(
            float(getattr(self.adaptive_config, "fixed_keep_ratio", 1.0))
            for _ in merge_indices
        )
        return {
            int(merge_index): self._fixed_group_keep_ratio_from_value(float(keep_ratio), layout)
            for merge_index, keep_ratio in zip(merge_indices, keep_ratios)
        }

    def _fixed_true_keep_ratios_by_stage(
        self,
        merge_indices: tuple[int, ...],
    ) -> dict[int, float]:
        raw_keep_ratios = tuple(
            float(value)
            for value in (getattr(self.adaptive_config, "fixed_keep_ratios", ()) or ())
        )
        if raw_keep_ratios and len(raw_keep_ratios) != len(merge_indices):
            raise ValueError(
                "fixed_keep_ratios must be empty or have the same length as merge stages: "
                f"got {len(raw_keep_ratios)} ratios for {len(merge_indices)} stages "
                f"(merge_indices={merge_indices})"
            )
        keep_ratios = raw_keep_ratios or tuple(
            float(getattr(self.adaptive_config, "fixed_keep_ratio", 1.0))
            for _ in merge_indices
        )
        return {
            int(merge_index): min(max(float(keep_ratio), 0.0), 1.0)
            for merge_index, keep_ratio in zip(merge_indices, keep_ratios)
        }

    def _true_token_keep_from_group_keep(self, group_keep_ratio: float, layout: GroupLayout) -> float:
        group_width = max(int(layout.max_group_size), 1)
        if group_width <= 1:
            return min(max(float(group_keep_ratio), 0.0), 1.0)
        group_keep_ratio = min(max(float(group_keep_ratio), 0.0), 1.0)
        return (1.0 + (group_width - 1) * group_keep_ratio) / float(group_width)

    def _batch_risk_complexity(
        self,
        group_scores: torch.Tensor,
        topk_ratio: float,
        num_frames: int | None = None,
    ) -> float | None:
        if group_scores.ndim != 2 or group_scores.shape[1] == 0:
            return None
        topk_ratio = min(max(topk_ratio, 0.0), 1.0)
        topk_groups = max(1, int(math.ceil(topk_ratio * group_scores.shape[1])))
        topk_groups = min(topk_groups, group_scores.shape[1])
        complexity = torch.sigmoid(torch.topk(group_scores, k=topk_groups, dim=1, largest=True).values).mean()
        temporal_scale = float(getattr(self.adaptive_config, "risk_dynamic_keep_temporal_scale", 0.0))
        if temporal_scale > 0.0 and num_frames is not None and num_frames > 1 and group_scores.shape[0] % num_frames == 0:
            grouped_scores = torch.sigmoid(group_scores).reshape(group_scores.shape[0] // num_frames, num_frames, -1)
            temporal_variation = torch.abs(grouped_scores[:, 1:] - grouped_scores[:, :-1]).mean()
            complexity = complexity + temporal_scale * temporal_variation
        complexity_value = float(complexity.detach().item())
        momentum = float(getattr(self.adaptive_config, "risk_dynamic_keep_momentum", 0.0))
        if momentum > 0.0:
            if self._dynamic_keep_complexity_ema is None:
                self._dynamic_keep_complexity_ema = complexity_value
            else:
                self._dynamic_keep_complexity_ema = (
                    momentum * self._dynamic_keep_complexity_ema
                    + (1.0 - momentum) * complexity_value
                )
            complexity_value = self._dynamic_keep_complexity_ema
        return float(min(max(complexity_value, 0.0), 1.0))

    def _adjust_ratio_from_complexity(
        self,
        base_ratio: float,
        complexity: float | None,
        reference: float,
        min_ratio: float,
        max_ratio: float,
        easy_scale: float,
        hard_scale: float,
        symmetric_scale: float,
    ) -> float:
        if complexity is None:
            return float(base_ratio)
        delta = float(complexity - reference)
        if delta >= 0.0:
            ratio = float(base_ratio + (hard_scale if hard_scale > 0.0 else symmetric_scale) * delta)
        else:
            ratio = float(base_ratio + (easy_scale if easy_scale > 0.0 else symmetric_scale) * delta)
        min_ratio = min(max(min_ratio, 0.0), 1.0)
        max_ratio = min(max(max_ratio, min_ratio), 1.0)
        # Keep dynamic adjustment centered around the requested base ratio instead of
        # letting an inconsistent config range override it completely.
        min_ratio = min(min_ratio, float(base_ratio))
        max_ratio = max(max_ratio, float(base_ratio))
        return min(max(ratio, min_ratio), max_ratio)

    def _adjust_batch_keep_ratio_from_risk(
        self,
        group_scores: torch.Tensor,
        base_keep_ratio: float,
        complexity: float | None = None,
        topk_ratio: float | None = None,
        num_frames: int | None = None,
    ) -> tuple[float, float | None]:
        if not bool(getattr(self.adaptive_config, "risk_dynamic_keep_enabled", False)):
            return float(base_keep_ratio), None
        keep_topk_ratio = float(
            getattr(self.adaptive_config, "risk_dynamic_keep_topk_ratio", 0.05) if topk_ratio is None else topk_ratio
        )
        if complexity is None:
            complexity = self._batch_risk_complexity(group_scores, topk_ratio=keep_topk_ratio, num_frames=num_frames)
        adjusted_keep_ratio = self._adjust_ratio_from_complexity(
            base_ratio=base_keep_ratio,
            complexity=complexity,
            reference=float(getattr(self.adaptive_config, "risk_dynamic_keep_reference", 0.5)),
            min_ratio=float(getattr(self.adaptive_config, "risk_dynamic_keep_min_ratio", 0.0)),
            max_ratio=float(getattr(self.adaptive_config, "risk_dynamic_keep_max_ratio", 1.0)),
            easy_scale=float(getattr(self.adaptive_config, "risk_dynamic_keep_easy_scale", 0.0)),
            hard_scale=float(getattr(self.adaptive_config, "risk_dynamic_keep_hard_scale", 0.0)),
            symmetric_scale=float(getattr(self.adaptive_config, "risk_dynamic_keep_scale", 0.15)),
        )
        previous_keep_ratio = self._dynamic_keep_ratio_ema
        step_limit = float(getattr(self.adaptive_config, "risk_dynamic_keep_step_limit", 0.0))
        momentum = float(getattr(self.adaptive_config, "risk_dynamic_keep_momentum", 0.0))
        if previous_keep_ratio is not None:
            if step_limit > 0.0:
                adjusted_keep_ratio = min(
                    max(adjusted_keep_ratio, previous_keep_ratio - step_limit),
                    previous_keep_ratio + step_limit,
                )
            if momentum > 0.0:
                adjusted_keep_ratio = momentum * previous_keep_ratio + (1.0 - momentum) * adjusted_keep_ratio
        self._dynamic_keep_ratio_ema = float(adjusted_keep_ratio)
        return adjusted_keep_ratio, complexity

    def _adjust_batch_hard_gate_ratio_from_risk(
        self,
        group_scores: torch.Tensor,
        base_hard_gate_ratio: float,
        complexity: float | None = None,
        topk_ratio: float | None = None,
    ) -> tuple[float, float | None]:
        if not bool(getattr(self.adaptive_config, "risk_dynamic_hard_gate_enabled", False)):
            return float(base_hard_gate_ratio), None
        hard_gate_topk_ratio = float(
            getattr(self.adaptive_config, "risk_dynamic_hard_gate_topk_ratio", 0.05)
            if topk_ratio is None
            else topk_ratio
        )
        if complexity is None:
            complexity = self._batch_risk_complexity(group_scores, topk_ratio=hard_gate_topk_ratio)
        adjusted_hard_gate_ratio = self._adjust_ratio_from_complexity(
            base_ratio=base_hard_gate_ratio,
            complexity=complexity,
            reference=float(getattr(self.adaptive_config, "risk_dynamic_hard_gate_reference", 0.5)),
            min_ratio=float(getattr(self.adaptive_config, "risk_dynamic_hard_gate_min_ratio", 0.0)),
            max_ratio=float(getattr(self.adaptive_config, "risk_dynamic_hard_gate_max_ratio", 1.0)),
            easy_scale=float(getattr(self.adaptive_config, "risk_dynamic_hard_gate_easy_scale", 0.0)),
            hard_scale=float(getattr(self.adaptive_config, "risk_dynamic_hard_gate_hard_scale", 0.0)),
            symmetric_scale=float(getattr(self.adaptive_config, "risk_dynamic_hard_gate_scale", 0.03)),
        )
        return adjusted_hard_gate_ratio, complexity

    def _prepare_risk_merge_stage(
        self,
        *,
        device: torch.device,
        patch_tokens: torch.Tensor,
        token_sizes_batch: torch.Tensor | None,
        layout: GroupLayout,
        layout_tensors: dict[str, torch.Tensor],
        frame_batch: torch.Tensor,
        patch_height: int,
        patch_width: int,
        batch_size: int,
        num_frames: int,
        keep_ratios: torch.Tensor,
        base_group_keep_ratio: float | None,
        merge_detail_profile: dict[str, float],
        collect_components: bool,
    ) -> dict[str, object]:
        merge_input_token_sizes = token_sizes_batch
        (
            (grouped_tokens, grouped_coords, member_mask, member_counts),
            group_pack_ms,
        ) = self._profile_section_ms(
            device,
            lambda: self._group_patch_tokens_and_coords(
                patch_tokens,
                layout,
                layout_tensors=layout_tensors,
            ),
        )
        merge_detail_profile["group_pack_ms"] += group_pack_ms
        ((group_scores, group_score_components), risk_score_ms) = self._profile_section_ms(
            device,
            lambda: self._compose_risk_v2_group_scores(
                grouped_tokens=grouped_tokens,
                grouped_coords=grouped_coords,
                member_mask=member_mask,
                member_counts=member_counts,
                patch_tokens=patch_tokens,
                layout=layout,
                frame_batch=frame_batch,
                patch_height=patch_height,
                patch_width=patch_width,
                batch_size=batch_size,
                num_frames=num_frames,
                collect_components=collect_components,
                all_members_valid=bool(layout_tensors.get("all_members_valid", False)),
            ),
        )
        merge_detail_profile["risk_score_ms"] += risk_score_ms
        detached_group_scores = group_scores.detach()
        risk_merge_strategy = str(getattr(self.adaptive_config, "risk_merge_strategy", "low_risk")).lower()
        candidate_needs_components = risk_merge_strategy in {"cam", "gradient", "grad"}
        ratio_policy_start = self._profile_marker(device)
        dynamic_keep_enabled = bool(getattr(self.adaptive_config, "risk_dynamic_keep_enabled", False))
        dynamic_hard_gate_enabled = bool(getattr(self.adaptive_config, "risk_dynamic_hard_gate_enabled", False))
        if dynamic_keep_enabled or dynamic_hard_gate_enabled:
            base_keep_ratio = float(keep_ratios.mean().detach().item())
        else:
            base_keep_ratio = (
                self._fixed_group_keep_ratio(layout)
                if base_group_keep_ratio is None
                else float(base_group_keep_ratio)
            )
        keep_topk_ratio = float(getattr(self.adaptive_config, "risk_dynamic_keep_topk_ratio", 0.05))
        hard_gate_topk_ratio = float(getattr(self.adaptive_config, "risk_dynamic_hard_gate_topk_ratio", 0.05))
        shared_complexity = None
        if (
            dynamic_keep_enabled
            and dynamic_hard_gate_enabled
            and math.isclose(keep_topk_ratio, hard_gate_topk_ratio)
        ):
            shared_complexity = self._batch_risk_complexity(
                detached_group_scores,
                topk_ratio=keep_topk_ratio,
                num_frames=num_frames,
            )
        keep_ratio, keep_complexity = self._adjust_batch_keep_ratio_from_risk(
            detached_group_scores,
            base_keep_ratio,
            complexity=shared_complexity,
            topk_ratio=keep_topk_ratio,
            num_frames=num_frames,
        )
        base_hard_gate_ratio = float(self.adaptive_config.risk_hard_gate_ratio)
        applied_hard_gate_ratio, hard_gate_complexity = self._adjust_batch_hard_gate_ratio_from_risk(
            detached_group_scores,
            base_hard_gate_ratio,
            complexity=shared_complexity,
            topk_ratio=hard_gate_topk_ratio,
        )
        requested_num_merge_groups = int((1.0 - keep_ratio) * layout.num_groups)
        requested_num_merge_groups = max(0, min(requested_num_merge_groups, max(layout.num_groups - 1, 0)))
        ratio_policy_end = self._profile_marker(device)
        merge_detail_profile["ratio_policy_ms"] += self._profile_elapsed_ms(ratio_policy_start, ratio_policy_end, device)
        has_protection = applied_hard_gate_ratio > 0.0
        if has_protection:
            (protected_masks, gate_select_ms) = self._profile_section_ms(
                device,
                lambda: self._risk_hard_gate_mask_batch(
                    detached_group_scores,
                    layout,
                    protect_ratio=applied_hard_gate_ratio,
                ),
            )
            candidate_num_merge_groups = None
        else:
            protected_masks = None
            gate_select_ms = 0.0
            candidate_num_merge_groups = requested_num_merge_groups
        (merged_group_masks, candidate_select_ms) = self._profile_section_ms(
            device,
            lambda: self._candidate_group_mask_batch_with_protection(
                group_scores=detached_group_scores,
                keep_ratio=keep_ratio,
                layout=layout,
                protected_mask=protected_masks,
                group_score_components=(
                    {key: value.detach() for key, value in group_score_components.items()}
                    if candidate_needs_components
                    else None
                ),
                num_merge_groups=candidate_num_merge_groups,
            ),
        )
        merge_detail_profile["gate_select_ms"] += gate_select_ms + candidate_select_ms
        actual_num_merge_groups = (
            requested_num_merge_groups
            if not has_protection
            else (int(merged_group_masks[0].sum().item()) if merged_group_masks.numel() > 0 else 0)
        )
        grouped_sizes_before_merge = None
        if merge_input_token_sizes is not None:
            grouped_sizes_before_merge, grouped_sizes_ms = self._profile_section_ms(
                device,
                lambda: merge_input_token_sizes.gather(
                    1,
                    layout_tensors["member_indices_clamped"].view(1, layout.num_groups * layout.max_group_size).expand(
                        patch_tokens.shape[0],
                        -1,
                    ),
                ).reshape(patch_tokens.shape[0], layout.num_groups, layout.max_group_size),
            )
            merge_detail_profile["grouped_sizes_ms"] += grouped_sizes_ms
        (
            (new_patch_tokens, merged_patch_coords_batch, merge_original_to_current, merged_token_sizes_batch),
            contract_merge_ms,
        ) = self._profile_section_ms(
            device,
            lambda: self._simple_merge_batch_with_mask(
                patch_tokens=patch_tokens,
                merged_group_mask=merged_group_masks,
                layout=layout,
                layout_tensors=layout_tensors,
                grouped_tokens=grouped_tokens,
                grouped_coords=grouped_coords,
                member_mask=member_mask,
                token_sizes=merge_input_token_sizes,
                grouped_sizes=grouped_sizes_before_merge,
                num_merge_groups=actual_num_merge_groups,
                return_token_sizes=True,
                profile=merge_detail_profile,
            ),
        )
        merge_detail_profile["contract_merge_ms"] += contract_merge_ms
        stage_merged_groups = int(actual_num_merge_groups * merged_group_masks.shape[0])
        stage_effective_merge = bool(stage_merged_groups > 0)
        if stage_effective_merge and self.token_transform_reducer is not None:
            (
                new_patch_tokens,
                token_transform_ms,
            ) = self._profile_section_ms(
                device,
                lambda: self._apply_token_transform_reducer(
                    patch_tokens=new_patch_tokens,
                    merged_group_mask=merged_group_masks,
                    grouped_tokens=grouped_tokens,
                    grouped_coords=grouped_coords,
                    member_mask=member_mask,
                    grouped_sizes=grouped_sizes_before_merge,
                    patch_hw=(patch_height, patch_width),
                ),
            )
            merge_detail_profile["contract_merge_ms"] += token_transform_ms
        correction_bias_batch = None
        if stage_effective_merge:
            (
                (new_patch_tokens, correction_bias_batch),
                correction_ms,
            ) = self._profile_section_ms(
                device,
                lambda: self._apply_merge_correction(
                    patch_tokens=new_patch_tokens,
                    patch_coords=merged_patch_coords_batch,
                    token_sizes=merged_token_sizes_batch,
                    patch_hw=(patch_height, patch_width),
                    merged_group_mask=merged_group_masks,
                    grouped_tokens=grouped_tokens,
                    grouped_coords=grouped_coords,
                    member_mask=member_mask,
                    grouped_sizes=grouped_sizes_before_merge,
                ),
            )
            merge_detail_profile["contract_merge_ms"] += correction_ms
        return {
            "group_scores": group_scores,
            "group_score_components": group_score_components,
            "grouped_tokens": grouped_tokens,
            "grouped_coords": grouped_coords,
            "member_mask": member_mask,
            "merged_group_masks": merged_group_masks,
            "protected_masks": protected_masks
            if protected_masks is not None
            else torch.zeros_like(merged_group_masks),
            "keep_ratio": float(keep_ratio),
            "true_token_keep_ratio": float(self._true_token_keep_from_group_keep(keep_ratio, layout)),
            "keep_complexity": keep_complexity,
            "base_hard_gate_ratio": float(base_hard_gate_ratio),
            "applied_hard_gate_ratio": float(applied_hard_gate_ratio),
            "hard_gate_complexity": hard_gate_complexity,
            "grouped_sizes_before_merge": grouped_sizes_before_merge,
            "new_patch_tokens": new_patch_tokens,
            "merged_patch_coords_batch": merged_patch_coords_batch,
            "merge_original_to_current": merge_original_to_current,
            "merged_token_sizes_batch": merged_token_sizes_batch,
            "correction_bias_batch": correction_bias_batch,
            "stage_merged_groups": int(stage_merged_groups),
            "stage_effective_merge": bool(stage_effective_merge),
            "stage_token_keep_ratio": float(new_patch_tokens.shape[1] / max(layout.patch_coords.shape[0], 1)),
        }

    def _record_risk_stage_output(
        self,
        *,
        stage_outputs: list[dict[str, object]],
        stage_state: dict[str, object],
        stage_type: str,
        block_index: int,
        layout: GroupLayout,
        patch_height: int,
        patch_width: int,
        num_frames_total: int,
        keep_ratios: torch.Tensor,
    ) -> None:
        group_score_components = stage_state["group_score_components"]
        group_scores = stage_state["group_scores"]
        merge_original_to_current = stage_state["merge_original_to_current"]
        protected_masks = stage_state["protected_masks"]
        merged_group_masks = stage_state["merged_group_masks"]
        stage_total_groups = int(num_frames_total * layout.num_groups)
        stage_protected_groups = int(protected_masks.sum().item())
        stage_candidate_groups = int(stage_total_groups - stage_protected_groups)
        frame_original_to_current = (
            [merge_original_to_current[frame_index].detach().clone() for frame_index in range(num_frames_total)]
            if bool(stage_state["stage_effective_merge"])
            else []
        )
        stage_outputs.append(
            {
                "merge_block_index": int(block_index),
                "merge_stage_type": stage_type,
                "patch_hw": (patch_height, patch_width),
                "frame_group_scores": [group_scores[frame_index] for frame_index in range(num_frames_total)],
                "frame_group_risk_prior": (
                    [group_score_components["risk_prior"][frame_index] for frame_index in range(num_frames_total)]
                    if "risk_prior" in group_score_components
                    else []
                ),
                "frame_group_compressibility": (
                    [group_score_components["compressibility"][frame_index] for frame_index in range(num_frames_total)]
                    if "compressibility" in group_score_components
                    else []
                ),
                "frame_group_task_preserve": (
                    [group_score_components["task_preserve"][frame_index] for frame_index in range(num_frames_total)]
                    if "task_preserve" in group_score_components
                    else []
                ),
                "frame_group_cam_activation": (
                    [group_score_components["cam_activation"][frame_index] for frame_index in range(num_frames_total)]
                    if "cam_activation" in group_score_components
                    else []
                ),
                "frame_group_image_gradient": (
                    [group_score_components["image_gradient"][frame_index] for frame_index in range(num_frames_total)]
                    if "image_gradient" in group_score_components
                    else []
                ),
                "frame_group_members": [layout.group_member_indices.clone() for _ in range(num_frames_total)],
                "frame_group_member_mask": [layout.group_member_mask.clone() for _ in range(num_frames_total)],
                "frame_merged_group_mask": [merged_group_masks[frame_index] for frame_index in range(num_frames_total)],
                "frame_group_protected_mask": [protected_masks[frame_index] for frame_index in range(num_frames_total)],
                "frame_original_to_current": frame_original_to_current,
                "keep_ratios": keep_ratios,
                "risk_complexity": float(stage_state["keep_complexity"] if stage_state["keep_complexity"] is not None else -1.0),
                "applied_keep_ratio": float(stage_state["keep_ratio"]),
                "applied_true_token_keep_ratio": float(stage_state["true_token_keep_ratio"]),
                "risk_merge_strategy": str(getattr(self.adaptive_config, "risk_merge_strategy", "low_risk")),
                "independent_compressibility_enabled": bool(self.group_compressibility_head is not None),
                "task_aware_preservation_enabled": bool(self.group_task_preserve_head is not None),
                "applied_hard_gate_ratio": float(stage_state["applied_hard_gate_ratio"]),
                "risk_hard_gate_ratio": float(stage_state["base_hard_gate_ratio"]),
                "effective_merge_applied": bool(stage_state["stage_effective_merge"]),
                "stage_merge_stats": {
                    "merge_block_index": int(block_index),
                    "merge_stage_type": stage_type,
                    "total_groups": int(stage_total_groups),
                    "candidate_groups": int(stage_candidate_groups),
                    "protected_groups": int(stage_protected_groups),
                    "merged_groups": int(stage_state["stage_merged_groups"]),
                    "effective_merge_applied": bool(stage_state["stage_effective_merge"]),
                    "risk_hard_gate_ratio": float(stage_state["base_hard_gate_ratio"]),
                    "applied_hard_gate_ratio": float(stage_state["applied_hard_gate_ratio"]),
                    "protected_ratio": float(stage_protected_groups / max(stage_total_groups, 1)),
                    "hard_gate_active": bool(stage_protected_groups > 0),
                    "applied_keep_ratio": float(stage_state["keep_ratio"]),
                    "applied_true_token_keep_ratio": float(stage_state["true_token_keep_ratio"]),
                    "risk_merge_strategy": str(getattr(self.adaptive_config, "risk_merge_strategy", "low_risk")),
                    "independent_compressibility_enabled": bool(self.group_compressibility_head is not None),
                    "task_aware_preservation_enabled": bool(self.group_task_preserve_head is not None),
                    "risk_complexity": float(
                        stage_state["keep_complexity"] if stage_state["keep_complexity"] is not None else -1.0
                    ),
                    "hard_gate_complexity": float(
                        stage_state["hard_gate_complexity"] if stage_state["hard_gate_complexity"] is not None else -1.0
                    ),
                    "token_keep_ratio": float(stage_state["stage_token_keep_ratio"]),
                },
            }
        )

    def _build_size_bias(self, token_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size_correction is None:
            zeros = token_sizes.new_zeros((token_sizes.shape[0], self.embed_dim))
            ones = token_sizes.new_ones((token_sizes.shape[0], self.embed_dim))
            return ones, zeros
        scale, bias = self.size_correction(token_sizes)
        scale = 1.0 + self.adaptive_config.size_correction_scale * (scale - 1.0)
        bias = self.adaptive_config.size_correction_scale * bias
        return scale, bias

    def _build_attention_bias(
        self,
        token_sizes: torch.Tensor | None,
        token_dtype: torch.dtype,
        extra_patch_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if token_sizes is None and extra_patch_bias is None:
            return None
        if not self.adaptive_config.use_attention_bias_correction and extra_patch_bias is None:
            return None
        if token_sizes is not None:
            patch_bias = torch.log(token_sizes.clamp_min(1.0)).to(dtype=token_dtype)
            if self.adaptive_config.attention_bias_correction_scale != 1.0:
                patch_bias = patch_bias * float(self.adaptive_config.attention_bias_correction_scale)
        else:
            patch_bias = extra_patch_bias.to(dtype=token_dtype)
        if extra_patch_bias is not None:
            patch_bias = patch_bias + float(self.adaptive_config.merge_correction_bias_scale) * extra_patch_bias.to(dtype=token_dtype)
        prefix_bias = torch.zeros(
            (patch_bias.shape[0], self.token_prefix),
            device=patch_bias.device,
            dtype=patch_bias.dtype,
        )
        key_bias = torch.cat((prefix_bias, patch_bias), dim=1)
        return key_bias[:, None, None, :]

    def _apply_merge_correction(
        self,
        patch_tokens: torch.Tensor,
        patch_coords: torch.Tensor | None,
        token_sizes: torch.Tensor | None,
        patch_hw: tuple[int, int],
        merged_group_mask: torch.Tensor | None = None,
        grouped_tokens: torch.Tensor | None = None,
        grouped_coords: torch.Tensor | None = None,
        member_mask: torch.Tensor | None = None,
        grouped_sizes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.merge_correction is None or patch_coords is None or token_sizes is None:
            return patch_tokens, None
        if (
            merged_group_mask is None
            or grouped_tokens is None
            or grouped_coords is None
            or member_mask is None
            or merged_group_mask.numel() == 0
        ):
            token_residual, attn_bias = self.merge_correction(
                patch_tokens=patch_tokens,
                patch_coords=patch_coords,
                token_sizes=token_sizes,
                patch_hw=patch_hw,
            )
            corrected_tokens = patch_tokens + float(self.adaptive_config.merge_correction_scale) * token_residual
            return corrected_tokens, attn_bias

        num_merge_groups = int(merged_group_mask[0].sum().item())
        if num_merge_groups == 0:
            return patch_tokens, patch_tokens.new_zeros(patch_tokens.shape[:2])

        num_frames_total, num_groups, group_width, channels = grouped_tokens.shape
        ordered_group_indices = self._ordered_group_indices_from_mask(merged_group_mask)
        token_order = ordered_group_indices.view(num_frames_total, num_groups, 1, 1).expand(-1, -1, group_width, channels)
        ordered_grouped_tokens = grouped_tokens.gather(1, token_order)[:, :num_merge_groups]

        coord_channels = grouped_coords.shape[-1]
        coord_order = ordered_group_indices.view(num_frames_total, num_groups, 1, 1).expand(
            -1,
            -1,
            group_width,
            coord_channels,
        )
        ordered_grouped_coords = grouped_coords.gather(1, coord_order)[:, :num_merge_groups]

        ordered_member_mask = member_mask.unsqueeze(0).expand(num_frames_total, -1, -1).gather(
            1,
            ordered_group_indices.unsqueeze(-1).expand(-1, -1, group_width),
        )[:, :num_merge_groups]

        ordered_grouped_sizes = None
        if grouped_sizes is not None:
            size_order = ordered_group_indices.view(num_frames_total, num_groups, 1).expand(-1, -1, group_width)
            ordered_grouped_sizes = grouped_sizes.gather(1, size_order)[:, :num_merge_groups]

        merged_patch_tokens = patch_tokens[:, :num_merge_groups]
        merged_patch_coords = patch_coords[:, :num_merge_groups]
        merged_token_sizes = token_sizes[:, :num_merge_groups]
        token_residual, attn_bias = self.merge_correction(
            patch_tokens=merged_patch_tokens,
            patch_coords=merged_patch_coords,
            token_sizes=merged_token_sizes,
            patch_hw=patch_hw,
            source_tokens=ordered_grouped_tokens,
            source_coords=ordered_grouped_coords,
            source_sizes=ordered_grouped_sizes,
            source_mask=ordered_member_mask,
        )
        corrected_tokens = patch_tokens.clone()
        corrected_tokens[:, :num_merge_groups] = (
            merged_patch_tokens + float(self.adaptive_config.merge_correction_scale) * token_residual
        )
        full_attn_bias = patch_tokens.new_zeros(patch_tokens.shape[:2])
        full_attn_bias[:, :num_merge_groups] = attn_bias
        return corrected_tokens, full_attn_bias

    def _apply_token_transform_reducer(
        self,
        patch_tokens: torch.Tensor,
        merged_group_mask: torch.Tensor,
        grouped_tokens: torch.Tensor,
        grouped_coords: torch.Tensor,
        member_mask: torch.Tensor,
        patch_hw: tuple[int, int],
        grouped_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.token_transform_reducer is None:
            return patch_tokens
        num_merge_groups = int(merged_group_mask[0].sum().item()) if merged_group_mask.numel() > 0 else 0
        if num_merge_groups == 0:
            return patch_tokens

        num_frames_total, num_groups, group_width, channels = grouped_tokens.shape
        ordered_group_indices = self._ordered_group_indices_from_mask(merged_group_mask)
        token_order = ordered_group_indices.view(num_frames_total, num_groups, 1, 1).expand(-1, -1, group_width, channels)
        ordered_grouped_tokens = grouped_tokens.gather(1, token_order)[:, :num_merge_groups]

        coord_channels = grouped_coords.shape[-1]
        coord_order = ordered_group_indices.view(num_frames_total, num_groups, 1, 1).expand(
            -1,
            -1,
            group_width,
            coord_channels,
        )
        ordered_grouped_coords = grouped_coords.gather(1, coord_order)[:, :num_merge_groups]

        ordered_member_mask = member_mask.unsqueeze(0).expand(num_frames_total, -1, -1).gather(
            1,
            ordered_group_indices.unsqueeze(-1).expand(-1, -1, group_width),
        )[:, :num_merge_groups]

        ordered_grouped_sizes = None
        if grouped_sizes is not None:
            size_order = ordered_group_indices.view(num_frames_total, num_groups, 1).expand(-1, -1, group_width)
            ordered_grouped_sizes = grouped_sizes.gather(1, size_order)[:, :num_merge_groups]

        transformed_tokens = self.token_transform_reducer(
            grouped_tokens=ordered_grouped_tokens,
            grouped_coords=ordered_grouped_coords,
            grouped_sizes=ordered_grouped_sizes,
            member_mask=ordered_member_mask,
            patch_hw=patch_hw,
        )
        refined_tokens = patch_tokens.clone()
        refined_tokens[:, :num_merge_groups] = transformed_tokens
        return refined_tokens

    def _apply_temporal_consensus_correction(
        self,
        patch_tokens: torch.Tensor,
        patch_coords: torch.Tensor | None,
        token_sizes: torch.Tensor | None,
        patch_hw: tuple[int, int],
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        if self.temporal_consensus_correction is None or patch_coords is None:
            return patch_tokens
        corrected_tokens = self.temporal_consensus_correction(
            patch_tokens=patch_tokens,
            patch_coords=patch_coords,
            token_sizes=token_sizes,
            patch_hw=patch_hw,
            batch_size=batch_size,
            num_frames=num_frames,
            stride=int(self.adaptive_config.temporal_consensus_stride),
        )
        scale = float(self.adaptive_config.temporal_consensus_scale)
        if scale == 1.0:
            return corrected_tokens
        return patch_tokens + scale * (corrected_tokens - patch_tokens)

    def _collect_existence_outputs(self, cls_tokens: list[torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
        if self.existence_head is None or not cls_tokens:
            return None
        pooled_cls = cls_tokens[-1].mean(dim=1)
        return {"existence_logits": self.existence_head(pooled_cls)}

    def _build_sparse_decoder_inputs(
        self,
        output_feature_maps: dict[int, torch.Tensor],
        merge_index: int,
        tokens: torch.Tensor,
        patch_height: int,
        patch_width: int,
        layout_tensors: dict[str, torch.Tensor],
        patch_coords_batch: torch.Tensor | None,
        token_sizes_batch: torch.Tensor | None,
        original_to_current: torch.Tensor | None,
        num_dense_sources: int,
        frame_batch: torch.Tensor,
        batch_size: int,
        num_frames: int,
        collect_profile: dict[str, float] | None = None,
    ) -> dict[str, object]:
        selected_indices: list[int] = []
        dense_feature_maps: list[torch.Tensor] = []
        if num_dense_sources > 0:
            dense_indices = [
                out_index
                for out_index in self.out_indices
                if out_index in output_feature_maps and out_index <= merge_index
            ]
            if not dense_indices:
                dense_indices = [out_index for out_index in self.out_indices if out_index in output_feature_maps]
            selected_indices = dense_indices[-num_dense_sources:]
            if len(selected_indices) == 1 and num_dense_sources == 2:
                selected_indices = [selected_indices[0], selected_indices[0]]
            dense_feature_maps = [output_feature_maps[index] for index in selected_indices]

        if collect_profile is not None:
            normalized, normalize_ms = self._profile_section_ms(tokens.device, lambda: self._normalize_output(tokens))
            collect_profile["collect_normalize_ms"] += normalize_ms
        else:
            normalized = self._normalize_output(tokens)
        sparse_patch_tokens = normalized[:, self.token_prefix :]
        if patch_coords_batch is None:
            sparse_patch_coords = layout_tensors["patch_coords"].unsqueeze(0).expand(sparse_patch_tokens.shape[0], -1, -1)
        else:
            sparse_patch_coords = patch_coords_batch
        if token_sizes_batch is None:
            sparse_token_sizes = self._runtime_ones(
                (sparse_patch_tokens.shape[0], sparse_patch_tokens.shape[1]),
                sparse_patch_tokens.device,
                sparse_patch_tokens.dtype,
            )
        else:
            sparse_token_sizes = token_sizes_batch.to(sparse_patch_tokens.dtype)
        if self.temporal_consensus_correction is not None:
            if collect_profile is not None:
                sparse_patch_tokens, temporal_ms = self._profile_section_ms(
                    sparse_patch_tokens.device,
                    lambda: self._apply_temporal_consensus_correction(
                        patch_tokens=sparse_patch_tokens,
                        patch_coords=sparse_patch_coords,
                        token_sizes=sparse_token_sizes,
                        patch_hw=(patch_height, patch_width),
                        batch_size=batch_size,
                        num_frames=num_frames,
                    ),
                )
                collect_profile["sparse_state_commit_ms"] += temporal_ms
            else:
                sparse_patch_tokens = self._apply_temporal_consensus_correction(
                    patch_tokens=sparse_patch_tokens,
                    patch_coords=sparse_patch_coords,
                    token_sizes=sparse_token_sizes,
                    patch_hw=(patch_height, patch_width),
                    batch_size=batch_size,
                    num_frames=num_frames,
                )
        restore_start = self._profile_marker(sparse_patch_tokens.device) if collect_profile is not None else None
        if self.decoder_variant == "decoder_token_pure":
            if original_to_current is None:
                restore_index = self._runtime_arange(
                    patch_height * patch_width,
                    sparse_patch_tokens.device,
                ).view(1, -1).expand(sparse_patch_tokens.shape[0], -1)
            else:
                restore_index = original_to_current
        else:
            restore_index = None
        needs_dark_prior = self.decoder_variant == "decoder_token_pure"
        if collect_profile is not None:
            restore_end = self._profile_marker(sparse_patch_tokens.device)
            collect_profile["restore_index_prep_ms"] += self._profile_elapsed_ms(
                restore_start,
                restore_end,
                sparse_patch_tokens.device,
            )
        if collect_profile is not None and needs_dark_prior:
            dark_prior_map, dark_prior_ms = self._profile_section_ms(
                sparse_patch_tokens.device,
                lambda: self._patch_dark_prior_map(
                    frame_batch,
                    patch_height=patch_height,
                    patch_width=patch_width,
                ),
            )
            collect_profile["dark_prior_ms"] += dark_prior_ms
        elif needs_dark_prior:
            dark_prior_map = self._patch_dark_prior_map(
                frame_batch,
                patch_height=patch_height,
                patch_width=patch_width,
            )
        else:
            dark_prior_map = None
        return {
            "dense_feature_maps": dense_feature_maps,
            "dense_source_indices": tuple(int(index) for index in selected_indices),
            "sparse_patch_tokens": sparse_patch_tokens,
            "sparse_patch_coords": sparse_patch_coords,
            "sparse_token_sizes": sparse_token_sizes,
            "restore_index": restore_index,
            "patch_hw": (patch_height, patch_width),
            "dark_prior_map": dark_prior_map,
        }

    def _forward_with_token_merging_risk_v1(self, clip: torch.Tensor, collect_adaptive_outputs: bool = True) -> dict[str, object]:
        frame_batch, batch_size, _, num_frames, _, _ = self._frame_batch(clip)
        device = frame_batch.device
        transient_merge_indices, persistent_merge_index = self._resolve_risk_v1_schedule()
        if self._should_hard_bypass_risk_v1():
            return self._forward_baseline(clip)
        merge_indices = tuple(sorted(transient_merge_indices + (persistent_merge_index,)))
        tail_stop_block_index = self._resolve_tail_stop_block_index(persistent_merge_index)
        if self.group_risk_head is None:
            raise RuntimeError("risk_v1 requires a group risk head.")

        pre_merge_start = self._profile_marker(device)
        pre_merge_end = None
        merge_start = None
        merge_end = None
        post_merge_start = None
        post_merge_end = None
        merge_detail_profile = {
            "group_pack_ms": 0.0,
            "risk_score_ms": 0.0,
            "gate_select_ms": 0.0,
            "contract_merge_ms": 0.0,
            "rope_rebuild_ms": 0.0,
            "post_blocks_ms": 0.0,
            "restore_collect_ms": 0.0,
            "collect_normalize_ms": 0.0,
            "restore_to_map_ms": 0.0,
            "collect_cls_ms": 0.0,
            "ratio_policy_ms": 0.0,
            "grouped_sizes_ms": 0.0,
            "merge_plan_ms": 0.0,
            "contract_kernel_ms": 0.0,
            "prefix_cat_ms": 0.0,
            "attention_bias_ms": 0.0,
            "size_bias_ms": 0.0,
            "merge_block_forward_ms": 0.0,
            "sparse_state_commit_ms": 0.0,
            "sparse_decoder_input_ms": 0.0,
            "restore_index_prep_ms": 0.0,
            "dark_prior_ms": 0.0,
            "post_size_bias_apply_ms": 0.0,
            "post_norm1_ms": 0.0,
            "post_qkv_ms": 0.0,
            "post_rope_apply_ms": 0.0,
            "post_attn_ms": 0.0,
            "post_attn_proj_ms": 0.0,
            "post_norm2_ms": 0.0,
            "post_mlp_fc1_ms": 0.0,
            "post_mlp_act_ms": 0.0,
            "post_mlp_fc2_ms": 0.0,
            "post_residual_ms": 0.0,
            "post_layout_ms": 0.0,
        }
        backbone_context = nullcontext if self.has_trainable_backbone else torch.no_grad
        with backbone_context():
            tokens, (patch_height, patch_width) = self.backbone.prepare_tokens_with_masks(frame_batch)

        base_rope = self._base_rope(patch_height, patch_width, tokens.device)
        layout = self._group_layout(patch_height, patch_width)
        layout_tensors = self._group_layout_tensors(layout, tokens.device)
        stage_group_keep_ratios = self._fixed_group_keep_ratios_by_stage(layout, merge_indices)
        requested_stage_keep_ratios = self._fixed_true_keep_ratios_by_stage(merge_indices)
        num_frames_total = frame_batch.shape[0]
        output_feature_maps: dict[int, torch.Tensor] = {}
        collect_cls_outputs = collect_adaptive_outputs and self.existence_head is not None
        output_cls_tokens: dict[int, torch.Tensor] = {} if collect_cls_outputs else {}
        stage_outputs: list[dict[str, object]] = []
        original_to_current: torch.Tensor | None = None
        patch_coords_batch: torch.Tensor | None = None
        token_sizes_batch: torch.Tensor | None = None
        attention_bias_batch: torch.Tensor | None = None
        cached_dynamic_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        cached_size_bias: tuple[torch.Tensor, torch.Tensor] | None = None
        fixed_group_keep_ratio = self._fixed_group_keep_ratio(layout)
        keep_ratios = tokens.new_full((batch_size,), fixed_group_keep_ratio)
        applied_keep_ratios_summary: tuple[float, ...] = tuple()
        requested_tail_stop_block_index = int(tail_stop_block_index)
        active_tail_stop_block_index = self.n_blocks - 1
        effective_merge_applied = False
        persistent_sparse_active = False
        post_block_indices: list[int] = []
        post_block_token_counts: list[int] = []

        for block_index, block in enumerate(self.backbone.blocks):
            stage_state: dict[str, object] | None = None
            stage_type: str | None = None
            stage_keep_ratios: torch.Tensor | None = None

            if not persistent_sparse_active and block_index in transient_merge_indices:
                stage_group_keep_ratio = stage_group_keep_ratios[int(block_index)]
                stage_keep_ratios = tokens.new_full((batch_size,), stage_group_keep_ratio)
                if pre_merge_end is None:
                    pre_merge_end = self._profile_marker(device)
                    merge_start = pre_merge_end
                stage_state = self._prepare_risk_merge_stage(
                    device=device,
                    patch_tokens=tokens[:, self.token_prefix :],
                    token_sizes_batch=None,
                    layout=layout,
                    layout_tensors=layout_tensors,
                    frame_batch=frame_batch,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    keep_ratios=stage_keep_ratios,
                    base_group_keep_ratio=stage_group_keep_ratio,
                    merge_detail_profile=merge_detail_profile,
                    collect_components=collect_adaptive_outputs,
                )
                applied_keep_ratios_summary = applied_keep_ratios_summary + (float(stage_state["keep_ratio"]),)
                if bool(stage_state["stage_effective_merge"]):
                    (
                        transient_rope,
                        rope_rebuild_ms,
                    ) = self._profile_section_ms(
                        device,
                        lambda: self._batched_rope_from_coords(
                            stage_state["merged_patch_coords_batch"],
                            patch_height,
                            patch_width,
                            dtype=self._runtime_rope_dtype(tokens),
                        ),
                    )
                    merge_detail_profile["rope_rebuild_ms"] += rope_rebuild_ms
                    transient_tokens, prefix_cat_ms = self._profile_section_ms(
                        device,
                        lambda: torch.cat(
                            (tokens[:, : self.token_prefix], stage_state["new_patch_tokens"]),
                            dim=1,
                        ),
                    )
                    merge_detail_profile["prefix_cat_ms"] += prefix_cat_ms
                    transient_attention_bias, attention_bias_ms = self._profile_section_ms(
                        device,
                        lambda: self._build_attention_bias(
                            stage_state["merged_token_sizes_batch"],
                            transient_tokens.dtype,
                            extra_patch_bias=stage_state["correction_bias_batch"],
                        ),
                    )
                    merge_detail_profile["attention_bias_ms"] += attention_bias_ms
                    transient_tokens, merge_block_forward_ms = self._profile_section_ms(
                        device,
                        lambda: self._run_block_batch(
                            block,
                            transient_tokens,
                            transient_rope,
                            token_sizes=stage_state["merged_token_sizes_batch"],
                            attn_bias=transient_attention_bias,
                        ),
                    )
                    merge_detail_profile["merge_block_forward_ms"] += merge_block_forward_ms
                    restored_patch_tokens = self._restore_patch_tokens_batch(
                        transient_tokens[:, self.token_prefix :],
                        stage_state["merge_original_to_current"],
                    )
                    tokens = torch.cat((transient_tokens[:, : self.token_prefix], restored_patch_tokens), dim=1)
                    stage_type = "transient"
                else:
                    tokens = self._run_block_batch(block, tokens, base_rope)
                    stage_type = "transient_skip"
                merge_end = self._profile_marker(device)
            elif not persistent_sparse_active and block_index == persistent_merge_index:
                if pre_merge_end is None:
                    pre_merge_end = self._profile_marker(device)
                    merge_start = pre_merge_end
                stage_group_keep_ratio = stage_group_keep_ratios[int(block_index)]
                stage_keep_ratios = tokens.new_full((batch_size,), stage_group_keep_ratio)
                stage_state = self._prepare_risk_merge_stage(
                    device=device,
                    patch_tokens=tokens[:, self.token_prefix :],
                    token_sizes_batch=None,
                    layout=layout,
                    layout_tensors=layout_tensors,
                    frame_batch=frame_batch,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    keep_ratios=stage_keep_ratios,
                    base_group_keep_ratio=stage_group_keep_ratio,
                    merge_detail_profile=merge_detail_profile,
                    collect_components=collect_adaptive_outputs,
                )
                applied_keep_ratios_summary = applied_keep_ratios_summary + (float(stage_state["keep_ratio"]),)
                use_persistent_sparse = bool(stage_state["stage_effective_merge"]) and self._should_commit_persistent_sparse_tail(
                    float(stage_state["keep_ratio"])
                )
                if use_persistent_sparse:
                    sparse_state_start = self._profile_marker(device)
                    patch_coords_batch = stage_state["merged_patch_coords_batch"]
                    original_to_current = stage_state["merge_original_to_current"]
                    token_sizes_batch = stage_state["merged_token_sizes_batch"]
                    sparse_state_end = self._profile_marker(device)
                    merge_detail_profile["sparse_state_commit_ms"] += self._profile_elapsed_ms(
                        sparse_state_start,
                        sparse_state_end,
                        device,
                    )
                    (
                        cached_dynamic_rope,
                        rope_rebuild_ms,
                    ) = self._profile_section_ms(
                        device,
                        lambda: self._batched_rope_from_coords(
                            stage_state["merged_patch_coords_batch"],
                            patch_height,
                            patch_width,
                            dtype=self._runtime_rope_dtype(tokens),
                        ),
                    )
                    merge_detail_profile["rope_rebuild_ms"] += rope_rebuild_ms
                    tokens, prefix_cat_ms = self._profile_section_ms(
                        device,
                        lambda: torch.cat((tokens[:, : self.token_prefix], stage_state["new_patch_tokens"]), dim=1),
                    )
                    merge_detail_profile["prefix_cat_ms"] += prefix_cat_ms
                    if token_sizes_batch is not None and self._size_bias_enabled():
                        cached_size_bias, size_bias_ms = self._profile_section_ms(
                            device,
                            lambda: self._build_size_bias(token_sizes_batch.to(tokens.dtype)),
                        )
                        merge_detail_profile["size_bias_ms"] += size_bias_ms
                    else:
                        cached_size_bias = None
                    attention_bias_batch, attention_bias_ms = self._profile_section_ms(
                        device,
                        lambda: self._build_attention_bias(
                            token_sizes_batch,
                            tokens.dtype,
                            extra_patch_bias=stage_state["correction_bias_batch"],
                        ),
                    )
                    merge_detail_profile["attention_bias_ms"] += attention_bias_ms
                    tokens, merge_block_forward_ms = self._profile_section_ms(
                        device,
                        lambda: self._run_compact_block_batch(
                            block,
                            tokens,
                            cached_dynamic_rope,
                            token_sizes=token_sizes_batch,
                            attn_bias=attention_bias_batch,
                            size_bias=cached_size_bias,
                        ),
                    )
                    merge_detail_profile["merge_block_forward_ms"] += merge_block_forward_ms
                    active_tail_stop_block_index = requested_tail_stop_block_index
                    persistent_sparse_active = True
                    effective_merge_applied = True
                    stage_type = "persistent_commit"
                elif bool(stage_state["stage_effective_merge"]) and bool(
                    getattr(self.adaptive_config, "risk_persistent_fallback_to_transient", True)
                ):
                    (
                        transient_rope,
                        rope_rebuild_ms,
                    ) = self._profile_section_ms(
                        device,
                        lambda: self._batched_rope_from_coords(
                            stage_state["merged_patch_coords_batch"],
                            patch_height,
                            patch_width,
                            dtype=self._runtime_rope_dtype(tokens),
                        ),
                    )
                    merge_detail_profile["rope_rebuild_ms"] += rope_rebuild_ms
                    transient_tokens, prefix_cat_ms = self._profile_section_ms(
                        device,
                        lambda: torch.cat(
                            (tokens[:, : self.token_prefix], stage_state["new_patch_tokens"]),
                            dim=1,
                        ),
                    )
                    merge_detail_profile["prefix_cat_ms"] += prefix_cat_ms
                    transient_attention_bias, attention_bias_ms = self._profile_section_ms(
                        device,
                        lambda: self._build_attention_bias(
                            stage_state["merged_token_sizes_batch"],
                            transient_tokens.dtype,
                            extra_patch_bias=stage_state["correction_bias_batch"],
                        ),
                    )
                    merge_detail_profile["attention_bias_ms"] += attention_bias_ms
                    transient_tokens, merge_block_forward_ms = self._profile_section_ms(
                        device,
                        lambda: self._run_block_batch(
                            block,
                            transient_tokens,
                            transient_rope,
                            token_sizes=stage_state["merged_token_sizes_batch"],
                            attn_bias=transient_attention_bias,
                        ),
                    )
                    merge_detail_profile["merge_block_forward_ms"] += merge_block_forward_ms
                    restored_patch_tokens = self._restore_patch_tokens_batch(
                        transient_tokens[:, self.token_prefix :],
                        stage_state["merge_original_to_current"],
                    )
                    tokens = torch.cat((transient_tokens[:, : self.token_prefix], restored_patch_tokens), dim=1)
                    stage_type = "persistent_transient"
                else:
                    tokens = self._run_block_batch(block, tokens, base_rope)
                    stage_type = "persistent_skip"
                merge_end = self._profile_marker(device)
                if effective_merge_applied and post_merge_start is None:
                    post_merge_start = merge_end
            else:
                rope = cached_dynamic_rope if cached_dynamic_rope is not None else base_rope
                if persistent_sparse_active and block_index > persistent_merge_index:
                    post_block_indices.append(int(block_index))
                    post_block_token_counts.append(int(tokens.shape[1] - self.token_prefix))
                if (
                    persistent_sparse_active
                    and block_index > persistent_merge_index
                    and self.enable_speed_profile
                    and self.enable_speed_profile_detail
                ):
                    def _run_post_block_detail():
                        return self._run_profiled_compact_block_batch(
                            block,
                            tokens,
                            rope,
                            token_sizes=token_sizes_batch,
                            attn_bias=attention_bias_batch,
                            size_bias=cached_size_bias,
                            profile=merge_detail_profile,
                        )
                    tokens, block_ms = self._profile_section_ms(device, _run_post_block_detail)
                    merge_detail_profile["post_blocks_ms"] += block_ms
                elif persistent_sparse_active and block_index > persistent_merge_index and self.enable_speed_profile:
                    def _run_post_block():
                        return self._run_compact_block_batch(
                            block,
                            tokens,
                            rope,
                            token_sizes=token_sizes_batch,
                            attn_bias=attention_bias_batch,
                            size_bias=cached_size_bias,
                        )
                    tokens, block_ms = self._profile_section_ms(device, _run_post_block)
                    merge_detail_profile["post_blocks_ms"] += block_ms
                elif persistent_sparse_active and block_index > persistent_merge_index:
                    tokens = self._run_compact_block_batch(
                        block,
                        tokens,
                        rope,
                        token_sizes=token_sizes_batch,
                        attn_bias=attention_bias_batch,
                        size_bias=cached_size_bias,
                    )
                else:
                    tokens = self._run_block_batch(
                        block,
                        tokens,
                        rope,
                        token_sizes=token_sizes_batch,
                        attn_bias=attention_bias_batch,
                        size_bias=cached_size_bias,
                    )

            should_collect_feature_map = False
            if block_index in self.out_indices:
                if not self._uses_sparse_decoder():
                    should_collect_feature_map = True
                elif self._decoder_requires_dense_sources():
                    # For very early persistent merge points (for example block 1),
                    # there may be no dense source levels at or before merge_index.
                    # In that case, keep collecting the first few later dense levels
                    # so decoder_simple still receives a valid pair of dense inputs.
                    should_collect_feature_map = (
                        block_index <= persistent_merge_index
                        or len(output_feature_maps) < int(getattr(self, "decoder_simple_num_dense_sources", 2))
                    )
            if should_collect_feature_map:
                if persistent_sparse_active and block_index > persistent_merge_index and self.enable_speed_profile and self.enable_speed_profile_detail:
                    collect_start = self._profile_marker(device)
                    feature_map, cls_tokens = self._collect_batch_output(
                        tokens=tokens,
                        original_to_current=original_to_current,
                        patch_height=patch_height,
                        patch_width=patch_width,
                        batch_size=batch_size,
                        num_frames=num_frames,
                        collect_cls=collect_cls_outputs,
                        collect_profile=merge_detail_profile,
                    )
                    collect_end = self._profile_marker(device)
                    merge_detail_profile["restore_collect_ms"] += self._profile_elapsed_ms(collect_start, collect_end, device)
                elif persistent_sparse_active and block_index > persistent_merge_index and self.enable_speed_profile:
                    def _collect_post_output():
                        return self._collect_batch_output(
                            tokens=tokens,
                            original_to_current=original_to_current,
                            patch_height=patch_height,
                            patch_width=patch_width,
                            batch_size=batch_size,
                            num_frames=num_frames,
                            collect_cls=collect_cls_outputs,
                        )
                    (feature_map, cls_tokens), collect_ms = self._profile_section_ms(device, _collect_post_output)
                    merge_detail_profile["restore_collect_ms"] += collect_ms
                else:
                    feature_map, cls_tokens = self._collect_batch_output(
                        tokens=tokens,
                        original_to_current=original_to_current,
                        patch_height=patch_height,
                        patch_width=patch_width,
                        batch_size=batch_size,
                        num_frames=num_frames,
                        collect_cls=collect_cls_outputs,
                    )
                output_feature_maps[block_index] = feature_map
                if collect_cls_outputs and cls_tokens is not None:
                    output_cls_tokens[block_index] = cls_tokens
            elif block_index in self.out_indices and collect_cls_outputs:
                normalized = self._normalize_output(tokens)
                output_cls_tokens[block_index] = normalized[:, 0].reshape(batch_size, num_frames, self.embed_dim)

            if stage_state is not None and collect_adaptive_outputs and stage_type is not None:
                self._record_risk_stage_output(
                    stage_outputs=stage_outputs,
                    stage_state=stage_state,
                    stage_type=stage_type,
                    block_index=block_index,
                    layout=layout,
                    patch_height=patch_height,
                    patch_width=patch_width,
                    num_frames_total=num_frames_total,
                    keep_ratios=stage_keep_ratios if stage_keep_ratios is not None else keep_ratios,
                )

            if block_index >= active_tail_stop_block_index:
                break

        post_merge_end = self._profile_marker(device)
        feature_maps = []
        cls_tokens = [output_cls_tokens[out_index] for out_index in sorted(output_cls_tokens)] if collect_cls_outputs else []
        sparse_decoder_inputs, sparse_decoder_input_ms = self._profile_section_ms(
            device,
            lambda: self._build_sparse_decoder_inputs(
                output_feature_maps=output_feature_maps,
                merge_index=persistent_merge_index,
                tokens=tokens,
                patch_height=patch_height,
                patch_width=patch_width,
                layout_tensors=layout_tensors,
                patch_coords_batch=patch_coords_batch,
                token_sizes_batch=token_sizes_batch,
                original_to_current=original_to_current,
                num_dense_sources=(int(getattr(self, "decoder_simple_num_dense_sources", 2)) if self.decoder_variant == "decoder_simple" else 0),
                frame_batch=frame_batch,
                batch_size=batch_size,
                num_frames=num_frames,
                collect_profile=merge_detail_profile if self.enable_speed_profile_detail else None,
            ),
        )
        merge_detail_profile["sparse_decoder_input_ms"] += sparse_decoder_input_ms

        final_token_keep_ratio = 1.0
        if effective_merge_applied and original_to_current is not None:
            final_token_keep_ratio = float(tokens[:, self.token_prefix :].shape[1] / max(layout.patch_coords.shape[0], 1))

        adaptive_outputs = None
        if collect_adaptive_outputs:
            adaptive_outputs = {
                "enabled": True,
                "runtime_mode": "risk_v1",
                "group_contract_extension_enabled": bool(self._group_contract_extension_enabled),
                "patch_hw": (patch_height, patch_width),
                "stage_outputs": stage_outputs,
                "keep_ratios_summary": applied_keep_ratios_summary if stage_outputs else tuple(),
                "requested_stage_keep_ratios": tuple(
                    requested_stage_keep_ratios[int(index)]
                    for index in merge_indices
                ),
                "requested_stage_group_keep_ratios": tuple(
                    stage_group_keep_ratios[int(index)]
                    for index in merge_indices
                ),
                "merge_block_indices": merge_indices,
                "transient_merge_block_indices": transient_merge_indices,
                "persistent_merge_block_index": int(persistent_merge_index),
                "tail_stop_block_index": int(active_tail_stop_block_index),
                "requested_tail_stop_block_index": int(requested_tail_stop_block_index),
                "skipped_tail_blocks": int(self.n_blocks - active_tail_stop_block_index - 1),
                "effective_merge_applied": bool(effective_merge_applied),
                "final_token_keep_ratio": final_token_keep_ratio,
                "risk_merge_strategy": str(getattr(self.adaptive_config, "risk_merge_strategy", "low_risk")),
                "fixed_keep_ratio_is_token_keep": bool(getattr(self.adaptive_config, "fixed_keep_ratio_is_token_keep", False)),
                "final_token_saved_ratio": float(1.0 - final_token_keep_ratio),
                "final_interface_token_keep_ratio": final_token_keep_ratio,
                "execution_granularity": "hybrid",
                "merge_semantics": "risk_v1_hybrid",
                "risk_hard_gate_ratio": float(self.adaptive_config.risk_hard_gate_ratio),
                "risk_dynamic_hard_gate_enabled": bool(self.adaptive_config.risk_dynamic_hard_gate_enabled),
                "risk_dynamic_keep_enabled": bool(self.adaptive_config.risk_dynamic_keep_enabled),
                "attention_bias_correction_enabled": bool(self.adaptive_config.use_attention_bias_correction),
                "merge_correction_enabled": bool(self.adaptive_config.use_merge_correction),
                "token_transform_reducer_enabled": bool(self.token_transform_reducer is not None),
                "temporal_consensus_correction_enabled": bool(self.temporal_consensus_correction is not None),
                "independent_compressibility_enabled": bool(self.group_compressibility_head is not None),
                "task_aware_preservation_enabled": bool(self.group_task_preserve_head is not None),
                "stage_merge_stats": tuple(stage["stage_merge_stats"] for stage in stage_outputs),
                "post_block_indices": tuple(post_block_indices),
                "post_block_token_counts": tuple(post_block_token_counts),
                "post_block_all_compact": bool(
                    not post_block_token_counts
                    or all(count == int(tokens[:, self.token_prefix :].shape[1]) for count in post_block_token_counts)
                ),
            }
        if self.enable_speed_profile:
            self.last_speed_profile = {
                "pre_merge_ms": self._profile_elapsed_ms(pre_merge_start, pre_merge_end, device),
                "merge_ms": self._profile_elapsed_ms(merge_start, merge_end, device),
                "post_merge_ms": self._profile_elapsed_ms(post_merge_start, post_merge_end, device),
                **merge_detail_profile,
            }
        return {
            "feature_maps": feature_maps,
            "cls_tokens": cls_tokens,
            "sparse_decoder_inputs": sparse_decoder_inputs,
            "adaptive_outputs": adaptive_outputs,
            "existence_outputs": self._collect_existence_outputs(cls_tokens) if collect_cls_outputs else None,
        }

    def forward(self, clip: torch.Tensor, collect_adaptive_outputs: bool = True) -> dict[str, object]:
        if not self.adaptive_config.enable_token_merging:
            return self._forward_baseline(clip)
        if not self.use_risk_v1_mode:
            raise ValueError(
                f"Only accuracy baseline and risk_v1 token merging are supported, got runtime_mode={self.runtime_mode!r}."
            )
        return self._forward_with_token_merging_risk_v1(clip, collect_adaptive_outputs=collect_adaptive_outputs)
