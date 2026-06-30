from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    return [_safe_int(value) for value in values]


def _to_shape_list(shape: Any) -> list[int]:
    if shape is None:
        return []
    return [_safe_int(value) for value in shape]


def _summarize_stage_stats(stage_stats: Any) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    if not stage_stats:
        return summarized
    for stage in stage_stats:
        summarized.append(
            {
                "merge_block_index": _safe_int(stage.get("merge_block_index")),
                "merge_stage_type": str(stage.get("merge_stage_type", "")),
                "total_groups": _safe_int(stage.get("total_groups")),
                "candidate_groups": _safe_int(stage.get("candidate_groups")),
                "protected_groups": _safe_int(stage.get("protected_groups")),
                "merged_groups": _safe_int(stage.get("merged_groups")),
                "effective_merge_applied": bool(stage.get("effective_merge_applied", False)),
                "risk_hard_gate_ratio": _safe_float(stage.get("risk_hard_gate_ratio")),
                "applied_hard_gate_ratio": _safe_float(stage.get("applied_hard_gate_ratio")),
                "protected_ratio": _safe_float(stage.get("protected_ratio")),
                "hard_gate_active": bool(stage.get("hard_gate_active", False)),
                "applied_keep_ratio": _safe_float(stage.get("applied_keep_ratio")),
                "applied_true_token_keep_ratio": _safe_float(
                    stage.get("applied_true_token_keep_ratio"), default=1.0
                ),
                "risk_complexity": _safe_float(stage.get("risk_complexity"), default=-1.0),
                "hard_gate_complexity": _safe_float(stage.get("hard_gate_complexity"), default=-1.0),
                "token_keep_ratio": _safe_float(stage.get("token_keep_ratio"), default=1.0),
            }
        )
    return summarized


def classify_merge_regime(summary: dict[str, Any]) -> str:
    requested_keep_ratio = _safe_float(summary.get("requested_keep_ratio"), default=1.0)
    final_token_keep_ratio = _safe_float(summary.get("final_token_keep_ratio"), default=1.0)
    effective_merge_applied = bool(summary.get("effective_merge_applied", False))
    stage_types = [str(stage.get("merge_stage_type", "")) for stage in summary.get("stage_merge_stats", [])]

    if requested_keep_ratio >= 0.999:
        if str(summary.get("runtime_mode", "")) == "baseline_sparse_decoder" or not summary.get("merge_block_indices"):
            return "baseline_keep_one_bypass"
        return "samepath_keep_one_no_merge"
    if any(stage_type == "persistent_commit" for stage_type in stage_types):
        return "persistent_sparse"
    if effective_merge_applied and final_token_keep_ratio < 0.999:
        return "persistent_sparse"
    if any(stage_type in {"transient", "persistent_transient"} for stage_type in stage_types):
        return "transient_only"
    if stage_types and all(stage_type.endswith("skip") for stage_type in stage_types):
        return "merge_scored_but_skipped"
    return "no_effective_merge"


def build_debug_summary(
    debug: dict[str, Any],
    *,
    requested_keep_ratio: float | None = None,
    seed: int | None = None,
    context: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_stats = _summarize_stage_stats(debug.get("stage_merge_stats"))
    sparse_tokens_shape = _to_shape_list(debug.get("sparse_decoder_sparse_tokens_shape"))
    sparse_token_count = sparse_tokens_shape[1] if len(sparse_tokens_shape) >= 2 else None
    sparse_token_dim = sparse_tokens_shape[2] if len(sparse_tokens_shape) >= 3 else None

    keep_ratios = debug.get("keep_ratios")
    inferred_requested_keep_ratio = None
    if keep_ratios:
        inferred_requested_keep_ratio = _safe_float(keep_ratios[0])

    summary: dict[str, Any] = {
        "context": context,
        "seed": seed,
        "runtime_mode": str(debug.get("runtime_mode", "")),
        "requested_keep_ratio": (
            _safe_float(requested_keep_ratio)
            if requested_keep_ratio is not None
            else (
                inferred_requested_keep_ratio
                if inferred_requested_keep_ratio is not None
                else None
            )
        ),
        "keep_ratios": [_safe_float(value) for value in keep_ratios] if keep_ratios else [],
        "requested_stage_keep_ratios": [
            _safe_float(value)
            for value in (debug.get("requested_stage_keep_ratios") or [])
        ],
        "requested_stage_group_keep_ratios": [
            _safe_float(value)
            for value in (debug.get("requested_stage_group_keep_ratios") or [])
        ],
        "merge_block_indices": _to_int_list(debug.get("merge_block_indices")),
        "transient_merge_block_indices": _to_int_list(debug.get("transient_merge_block_indices")),
        "persistent_merge_block_index": (
            None
            if debug.get("persistent_merge_block_index") is None
            else _safe_int(debug.get("persistent_merge_block_index"))
        ),
        "tail_stop_block_index": _safe_int(debug.get("tail_stop_block_index"), default=-1),
        "requested_tail_stop_block_index": _safe_int(debug.get("requested_tail_stop_block_index"), default=-1),
        "skipped_tail_blocks": _safe_int(debug.get("skipped_tail_blocks")),
        "effective_merge_applied": bool(debug.get("effective_merge_applied", False)),
        "execution_granularity": str(debug.get("execution_granularity", "")),
        "merge_semantics": str(debug.get("merge_semantics", "")),
        "risk_merge_strategy": str(debug.get("risk_merge_strategy", "")),
        "final_token_keep_ratio": _safe_float(debug.get("final_token_keep_ratio"), default=1.0),
        "final_token_saved_ratio": _safe_float(debug.get("final_token_saved_ratio"), default=0.0),
        "final_interface_token_keep_ratio": _safe_float(
            debug.get("final_interface_token_keep_ratio"), default=1.0
        ),
        "dense_source_indices": _to_int_list(debug.get("sparse_decoder_dense_source_indices")),
        "dense_source_count": len(debug.get("sparse_decoder_dense_source_indices", []) or []),
        "sparse_patch_hw": _to_shape_list(debug.get("sparse_decoder_patch_hw")),
        "sparse_tokens_shape": sparse_tokens_shape,
        "sparse_token_count": sparse_token_count,
        "sparse_token_dim": sparse_token_dim,
        "decoder_variant": str(debug.get("decoder_variant", "")),
        "decoder_simple_num_dense_sources": (
            None
            if debug.get("decoder_simple_num_dense_sources") is None
            else _safe_int(debug.get("decoder_simple_num_dense_sources"))
        ),
        "group_contract_extension_enabled": bool(debug.get("group_contract_extension_enabled", False)),
        "attention_bias_correction_enabled": bool(debug.get("attention_bias_correction_enabled", False)),
        "merge_correction_enabled": bool(debug.get("merge_correction_enabled", False)),
        "token_transform_reducer_enabled": bool(debug.get("token_transform_reducer_enabled", False)),
        "temporal_consensus_correction_enabled": bool(debug.get("temporal_consensus_correction_enabled", False)),
        "risk_hard_gate_ratio": _safe_float(debug.get("risk_hard_gate_ratio")),
        "risk_dynamic_keep_enabled": bool(debug.get("risk_dynamic_keep_enabled", False)),
        "risk_dynamic_hard_gate_enabled": bool(debug.get("risk_dynamic_hard_gate_enabled", False)),
        "stage_merge_stats": stage_stats,
        "post_block_indices": _to_int_list(debug.get("post_block_indices")),
        "post_block_token_counts": _to_int_list(debug.get("post_block_token_counts")),
        "post_block_all_compact": bool(debug.get("post_block_all_compact", False)),
    }
    summary["regime"] = classify_merge_regime(summary)
    if extra:
        summary.update(extra)
    return summary


def save_debug_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
