from __future__ import annotations

import importlib
from typing import Any

import torch

_EXT_MODULE: Any | None = None
_EXT_IMPORT_ERROR: Exception | None = None


def _load_extension() -> Any | None:
    global _EXT_MODULE, _EXT_IMPORT_ERROR
    if _EXT_MODULE is not None:
        return _EXT_MODULE
    if _EXT_IMPORT_ERROR is not None:
        return None
    try:
        _EXT_MODULE = importlib.import_module(".ext", package=__name__)
        return _EXT_MODULE
    except Exception as exc:  # pragma: no cover - optional runtime path
        _EXT_IMPORT_ERROR = exc
        return None


def group_contract_available() -> bool:
    return _load_extension() is not None


def group_contract_import_error() -> Exception | None:
    _load_extension()
    return _EXT_IMPORT_ERROR


def _group_contract_fallback(
    grouped_tokens: torch.Tensor,
    dst_index: torch.Tensor,
    out_rows: int,
) -> torch.Tensor:
    if grouped_tokens.ndim != 4:
        raise ValueError(f"Expected grouped_tokens [B, N, G, C], got {tuple(grouped_tokens.shape)}")
    if dst_index.ndim != 2:
        raise ValueError(f"Expected dst_index [B, N*G], got {tuple(dst_index.shape)}")
    batch_size, num_groups, group_width, channels = grouped_tokens.shape
    flat_tokens = grouped_tokens.reshape(batch_size, num_groups * group_width, channels)
    out = grouped_tokens.new_zeros((batch_size, out_rows, channels))
    scatter_index = dst_index.unsqueeze(-1).expand(-1, -1, channels)
    out.scatter_add_(1, scatter_index, flat_tokens)
    counts = grouped_tokens.new_zeros((batch_size, out_rows, 1))
    ones = grouped_tokens.new_ones((batch_size, num_groups * group_width, 1))
    counts.scatter_add_(1, dst_index.unsqueeze(-1), ones)
    return out / counts.clamp_min(1.0)


def _group_contract_unweighted_with_sizes_fallback(
    grouped_tokens: torch.Tensor,
    grouped_coords: torch.Tensor,
    dst_index: torch.Tensor,
    out_rows: int,
    num_merge_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    merged_tokens = _group_contract_fallback(grouped_tokens, dst_index, out_rows)
    merged_coords = _group_contract_fallback(grouped_coords, dst_index, out_rows)
    token_sizes = grouped_tokens.new_ones((grouped_tokens.shape[0], int(out_rows)))
    if num_merge_groups > 0:
        token_sizes[:, : int(num_merge_groups)] = float(grouped_tokens.shape[2])
    return merged_tokens, merged_coords, token_sizes


def _group_contract_merge_mask_unweighted_with_sizes_fallback(
    grouped_tokens: torch.Tensor,
    grouped_coords: torch.Tensor,
    merged_group_mask: torch.Tensor,
    member_indices: torch.Tensor,
    out_rows: int,
    num_merge_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group_width = int(grouped_tokens.shape[2])
    dst_index, old_to_new = _group_contract_plan_fallback(
        merged_group_mask,
        member_indices,
        group_width,
        int(num_merge_groups),
    )
    merged_tokens, merged_coords, token_sizes = _group_contract_unweighted_with_sizes_fallback(
        grouped_tokens,
        grouped_coords,
        dst_index,
        int(out_rows),
        int(num_merge_groups),
    )
    return merged_tokens, merged_coords, old_to_new, token_sizes


def _group_contract_scan_fallback(
    merged_group_mask: torch.Tensor,
    group_width: int,
) -> torch.Tensor:
    if merged_group_mask.ndim != 2:
        raise ValueError(f"Expected merged_group_mask [B, N], got {tuple(merged_group_mask.shape)}")
    if group_width <= 0:
        raise ValueError(f"group_width must be positive, got {group_width}")
    mask = merged_group_mask.to(dtype=torch.bool)
    merged_ranks = torch.cumsum(mask.long(), dim=1) - 1
    kept_mask = ~mask
    kept_ranks = torch.cumsum(kept_mask.long(), dim=1) - 1
    num_merge_groups = mask.long().sum(dim=1, keepdim=True)
    member_offsets = torch.arange(group_width, device=mask.device, dtype=torch.long).view(1, 1, group_width)
    dst_index = torch.where(
        mask.unsqueeze(-1),
        merged_ranks.unsqueeze(-1),
        num_merge_groups.unsqueeze(-1) + kept_ranks.unsqueeze(-1) * group_width + member_offsets,
    )
    return dst_index.reshape(mask.shape[0], mask.shape[1] * group_width)


def _group_contract_plan_fallback(
    merged_group_mask: torch.Tensor,
    member_indices: torch.Tensor,
    group_width: int,
    num_merge_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dst_index = _group_contract_scan_fallback(merged_group_mask, int(group_width))
    batch_size = int(dst_index.shape[0])
    flat_members = member_indices.reshape(1, -1).expand(batch_size, -1)
    old_to_new = torch.empty_like(dst_index)
    old_to_new.scatter_(1, flat_members, dst_index)
    return dst_index, old_to_new


def _group_restore_fallback(
    patch_tokens: torch.Tensor,
    old_to_new: torch.Tensor,
) -> torch.Tensor:
    if patch_tokens.ndim != 3:
        raise ValueError(f"Expected patch_tokens [B, M, C], got {tuple(patch_tokens.shape)}")
    if old_to_new.ndim != 2:
        raise ValueError(f"Expected old_to_new [B, N], got {tuple(old_to_new.shape)}")
    if patch_tokens.shape[0] != old_to_new.shape[0]:
        raise ValueError("Batch size mismatch between patch_tokens and old_to_new")
    gather_index = old_to_new.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
    return patch_tokens.gather(1, gather_index)


def _group_restore_to_map_fallback(
    patch_tokens: torch.Tensor,
    old_to_new: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    restored = _group_restore_fallback(patch_tokens, old_to_new)
    expected_tokens = patch_height * patch_width
    if restored.shape[1] != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} original tokens for patch map, got {restored.shape[1]}"
        )
    return restored.reshape(restored.shape[0], patch_height, patch_width, restored.shape[-1]).permute(0, 3, 1, 2).contiguous()


def group_contract_merge(
    grouped_tokens: torch.Tensor,
    dst_index: torch.Tensor,
    out_rows: int,
) -> torch.Tensor:
    ext = _load_extension()
    if ext is not None and grouped_tokens.is_cuda and dst_index.is_cuda:
        return ext.group_contract_merge(grouped_tokens.contiguous(), dst_index.contiguous(), int(out_rows))
    return _group_contract_fallback(grouped_tokens, dst_index, int(out_rows))


def group_contract_merge_unweighted_with_sizes(
    grouped_tokens: torch.Tensor,
    grouped_coords: torch.Tensor,
    dst_index: torch.Tensor,
    out_rows: int,
    num_merge_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    if ext is not None and grouped_tokens.is_cuda and grouped_coords.is_cuda and dst_index.is_cuda:
        merged_tokens, merged_coords, token_sizes = ext.group_contract_merge_unweighted_with_sizes(
            grouped_tokens.contiguous(),
            grouped_coords.contiguous(),
            dst_index.contiguous(),
            int(out_rows),
            int(num_merge_groups),
        )
        return merged_tokens, merged_coords, token_sizes
    return _group_contract_unweighted_with_sizes_fallback(
        grouped_tokens,
        grouped_coords,
        dst_index,
        int(out_rows),
        int(num_merge_groups),
    )


def group_contract_merge_mask_unweighted_with_sizes(
    grouped_tokens: torch.Tensor,
    grouped_coords: torch.Tensor,
    merged_group_mask: torch.Tensor,
    member_indices: torch.Tensor,
    out_rows: int,
    num_merge_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    if (
        ext is not None
        and grouped_tokens.is_cuda
        and grouped_coords.is_cuda
        and merged_group_mask.is_cuda
        and member_indices.is_cuda
    ):
        merged_tokens, merged_coords, old_to_new, token_sizes = ext.group_contract_merge_mask_unweighted_with_sizes(
            grouped_tokens.contiguous(),
            grouped_coords.contiguous(),
            merged_group_mask.to(dtype=torch.bool).contiguous(),
            member_indices.contiguous(),
            int(out_rows),
            int(num_merge_groups),
        )
        return merged_tokens, merged_coords, old_to_new, token_sizes
    return _group_contract_merge_mask_unweighted_with_sizes_fallback(
        grouped_tokens,
        grouped_coords,
        merged_group_mask,
        member_indices,
        int(out_rows),
        int(num_merge_groups),
    )


def group_contract_scan(
    merged_group_mask: torch.Tensor,
    group_width: int,
) -> torch.Tensor:
    ext = _load_extension()
    if ext is not None:
        return ext.group_contract_scan(merged_group_mask.contiguous(), int(group_width))
    return _group_contract_scan_fallback(merged_group_mask, int(group_width))


def group_contract_plan(
    merged_group_mask: torch.Tensor,
    member_indices: torch.Tensor,
    group_width: int,
    num_merge_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ext = _load_extension()
    if ext is not None and merged_group_mask.is_cuda and member_indices.is_cuda:
        dst_index, old_to_new = ext.group_contract_plan(
            merged_group_mask.contiguous(),
            member_indices.contiguous(),
            int(group_width),
            int(num_merge_groups),
        )
        return dst_index, old_to_new
    return _group_contract_plan_fallback(
        merged_group_mask,
        member_indices,
        int(group_width),
        int(num_merge_groups),
    )


def group_restore(
    patch_tokens: torch.Tensor,
    old_to_new: torch.Tensor,
) -> torch.Tensor:
    ext = _load_extension()
    if ext is not None and patch_tokens.is_cuda and old_to_new.is_cuda:
        return ext.group_restore(patch_tokens.contiguous(), old_to_new.contiguous())
    return _group_restore_fallback(patch_tokens, old_to_new)


def group_restore_to_map(
    patch_tokens: torch.Tensor,
    old_to_new: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> torch.Tensor:
    ext = _load_extension()
    if ext is not None and patch_tokens.is_cuda and old_to_new.is_cuda:
        return ext.group_restore_to_map(
            patch_tokens.contiguous(),
            old_to_new.contiguous(),
            int(patch_height),
            int(patch_width),
        )
    return _group_restore_to_map_fallback(patch_tokens, old_to_new, int(patch_height), int(patch_width))


__all__ = [
    "group_contract_available",
    "group_contract_import_error",
    "group_contract_merge",
    "group_contract_merge_mask_unweighted_with_sizes",
    "group_contract_merge_unweighted_with_sizes",
    "group_contract_plan",
    "group_contract_scan",
    "group_restore",
    "group_restore_to_map",
]
