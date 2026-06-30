import copy
import json
from pathlib import Path

from rvsd.configs.baseline import RVSDBaselineConfig
from rvsd.tools.debug_diagnostics import build_debug_summary, save_debug_summary
from rvsd.tools.infer import capture_first_debug_snapshot, measure_model_speed_stats


def _real_speed_config(config: RVSDBaselineConfig) -> RVSDBaselineConfig:
    speed_config = copy.deepcopy(config)
    speed_config.dataset.dummy_mode = False
    speed_config.run.batch_size = 1
    speed_config.run.save_predictions = False
    speed_config.run.save_visualizations = False
    return speed_config


def run_real_speed_test(config: RVSDBaselineConfig, split: str = "test") -> dict[str, float | int | str]:
    speed_config = _real_speed_config(config)
    speed_stats = measure_model_speed_stats(speed_config, split=split)

    metrics: dict[str, float | int | str] = {
        "measurement_type": "real_data_model_fps",
        "model_fps": float(speed_stats["model_fps"]),
        "mean_iter_ms": float(speed_stats["mean_iter_ms"]),
        "avg_pre_merge_ms": float(speed_stats["avg_pre_merge_ms"]),
        "avg_merge_ms": float(speed_stats["avg_merge_ms"]),
        "avg_post_merge_ms": float(speed_stats["avg_post_merge_ms"]),
        "avg_decoder_head_ms": float(speed_stats["avg_decoder_head_ms"]),
        "avg_group_pack_ms": float(speed_stats.get("avg_group_pack_ms", 0.0)),
        "avg_risk_score_ms": float(speed_stats.get("avg_risk_score_ms", 0.0)),
        "avg_gate_select_ms": float(speed_stats.get("avg_gate_select_ms", 0.0)),
        "avg_contract_merge_ms": float(speed_stats.get("avg_contract_merge_ms", 0.0)),
        "avg_rope_rebuild_ms": float(speed_stats.get("avg_rope_rebuild_ms", 0.0)),
        "avg_post_blocks_ms": float(speed_stats.get("avg_post_blocks_ms", 0.0)),
        "avg_restore_collect_ms": float(speed_stats.get("avg_restore_collect_ms", 0.0)),
        "avg_collect_normalize_ms": float(speed_stats.get("avg_collect_normalize_ms", 0.0)),
        "avg_restore_to_map_ms": float(speed_stats.get("avg_restore_to_map_ms", 0.0)),
        "avg_collect_cls_ms": float(speed_stats.get("avg_collect_cls_ms", 0.0)),
        "avg_ratio_policy_ms": float(speed_stats.get("avg_ratio_policy_ms", 0.0)),
        "avg_grouped_sizes_ms": float(speed_stats.get("avg_grouped_sizes_ms", 0.0)),
        "avg_merge_plan_ms": float(speed_stats.get("avg_merge_plan_ms", 0.0)),
        "avg_contract_kernel_ms": float(speed_stats.get("avg_contract_kernel_ms", 0.0)),
        "avg_prefix_cat_ms": float(speed_stats.get("avg_prefix_cat_ms", 0.0)),
        "avg_attention_bias_ms": float(speed_stats.get("avg_attention_bias_ms", 0.0)),
        "avg_size_bias_ms": float(speed_stats.get("avg_size_bias_ms", 0.0)),
        "avg_merge_block_forward_ms": float(speed_stats.get("avg_merge_block_forward_ms", 0.0)),
        "avg_sparse_state_commit_ms": float(speed_stats.get("avg_sparse_state_commit_ms", 0.0)),
        "avg_sparse_decoder_input_ms": float(speed_stats.get("avg_sparse_decoder_input_ms", 0.0)),
        "avg_restore_index_prep_ms": float(speed_stats.get("avg_restore_index_prep_ms", 0.0)),
        "avg_dark_prior_ms": float(speed_stats.get("avg_dark_prior_ms", 0.0)),
        "avg_post_size_bias_apply_ms": float(speed_stats.get("avg_post_size_bias_apply_ms", 0.0)),
        "avg_post_norm1_ms": float(speed_stats.get("avg_post_norm1_ms", 0.0)),
        "avg_post_qkv_ms": float(speed_stats.get("avg_post_qkv_ms", 0.0)),
        "avg_post_rope_apply_ms": float(speed_stats.get("avg_post_rope_apply_ms", 0.0)),
        "avg_post_attn_ms": float(speed_stats.get("avg_post_attn_ms", 0.0)),
        "avg_post_attn_proj_ms": float(speed_stats.get("avg_post_attn_proj_ms", 0.0)),
        "avg_post_norm2_ms": float(speed_stats.get("avg_post_norm2_ms", 0.0)),
        "avg_post_mlp_fc1_ms": float(speed_stats.get("avg_post_mlp_fc1_ms", 0.0)),
        "avg_post_mlp_act_ms": float(speed_stats.get("avg_post_mlp_act_ms", 0.0)),
        "avg_post_mlp_fc2_ms": float(speed_stats.get("avg_post_mlp_fc2_ms", 0.0)),
        "avg_post_residual_ms": float(speed_stats.get("avg_post_residual_ms", 0.0)),
        "avg_post_layout_ms": float(speed_stats.get("avg_post_layout_ms", 0.0)),
        "frames_per_iter": int(speed_stats["frames_per_iter"]),
        "measured_iterations": int(speed_stats["measured_iterations"]),
        "measured_batch_shape": str(tuple(speed_stats["measured_batch_shape"])),
        "batch_size": int(speed_config.run.batch_size),
        "clip_length": int(speed_config.dataset.clip_length),
        "image_height": int(speed_config.dataset.image_size[0]),
        "image_width": int(speed_config.dataset.image_size[1]),
        "requested_keep_ratio": float(speed_config.model.adaptive.fixed_keep_ratio),
        "requested_keep_ratios": [
            float(value)
            for value in (getattr(speed_config.model.adaptive, "fixed_keep_ratios", ()) or ())
        ],
        "fixed_keep_ratio_is_token_keep": str(bool(speed_config.model.adaptive.fixed_keep_ratio_is_token_keep)),
        "fps_warmup_steps": int(speed_config.run.fps_warmup_steps),
        "fps_measure_steps": int(speed_config.run.fps_measure_steps),
        "speed_profile_enabled": str(bool(getattr(speed_config.run, "speed_profile_enabled", False))),
        "cuda_graph_requested": str(bool(speed_stats.get("cuda_graph_requested", False))),
        "cuda_graph_used": str(bool(speed_stats.get("cuda_graph_used", False))),
        "cuda_graph_disable_group_contract_extension": str(
            bool(getattr(speed_config.run, "cuda_graph_disable_group_contract_extension", True))
        ),
        "cuda_graph_error": str(speed_stats.get("cuda_graph_error", "")),
        "compile_inference": str(bool(getattr(speed_config.run, "compile_inference", False))),
        "compile_backbone_blocks": str(bool(getattr(speed_config.run, "compile_backbone_blocks", False))),
        "compile_backbone_block_indices": [
            int(index) for index in (getattr(speed_config.run, "compile_backbone_block_indices", ()) or ())
        ],
        "split": split,
        "dummy_mode": str(speed_config.dataset.dummy_mode),
    }

    output_dir = Path(speed_config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "speed_real_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    debug_snapshot = capture_first_debug_snapshot(speed_config, split=split)
    if debug_snapshot is not None:
        debug_summary = build_debug_summary(
            debug_snapshot,
            requested_keep_ratio=float(speed_config.model.adaptive.fixed_keep_ratio),
            seed=int(speed_config.run.seed),
            context="speed_real_probe",
            extra={
                "split": split,
                "output_dir": str(output_dir),
                "speed_metrics": metrics,
            },
        )
        diagnostics_path = save_debug_summary(output_dir / "speed_real_debug_summary.json", debug_summary)
        print(f"[speed_real] debug summary saved to {diagnostics_path}")

    print("[speed_real] real-data, batch=1, model-only FPS measurement")
    print(
        "[speed_real] "
        f"split={split} "
        f"batch_size={speed_config.run.batch_size} "
        f"clip_length={speed_config.dataset.clip_length} "
        f"image_size={tuple(speed_config.dataset.image_size)} "
        f"warmup={speed_config.run.fps_warmup_steps} "
        f"measure={speed_config.run.fps_measure_steps} "
        f"profile={bool(getattr(speed_config.run, 'speed_profile_enabled', False))} "
        f"cuda_graph={bool(speed_stats.get('cuda_graph_used', False))} "
        f"compile_blocks={bool(getattr(speed_config.run, 'compile_backbone_blocks', False))} "
        f"frames_per_iter={metrics['frames_per_iter']} "
        f"mean_iter_ms={metrics['mean_iter_ms']:.3f} "
        f"pre_merge_ms={metrics['avg_pre_merge_ms']:.3f} "
        f"merge_ms={metrics['avg_merge_ms']:.3f} "
        f"post_merge_ms={metrics['avg_post_merge_ms']:.3f} "
        f"decoder_head_ms={metrics['avg_decoder_head_ms']:.3f} "
        f"group_pack_ms={metrics['avg_group_pack_ms']:.3f} "
        f"risk_score_ms={metrics['avg_risk_score_ms']:.3f} "
        f"gate_select_ms={metrics['avg_gate_select_ms']:.3f} "
        f"contract_merge_ms={metrics['avg_contract_merge_ms']:.3f} "
        f"rope_rebuild_ms={metrics['avg_rope_rebuild_ms']:.3f} "
        f"post_blocks_ms={metrics['avg_post_blocks_ms']:.3f} "
        f"restore_collect_ms={metrics['avg_restore_collect_ms']:.3f} "
        f"collect_normalize_ms={metrics['avg_collect_normalize_ms']:.3f} "
        f"restore_to_map_ms={metrics['avg_restore_to_map_ms']:.3f} "
        f"collect_cls_ms={metrics['avg_collect_cls_ms']:.3f} "
        f"ratio_policy_ms={metrics['avg_ratio_policy_ms']:.3f} "
        f"grouped_sizes_ms={metrics['avg_grouped_sizes_ms']:.3f} "
        f"merge_plan_ms={metrics['avg_merge_plan_ms']:.3f} "
        f"contract_kernel_ms={metrics['avg_contract_kernel_ms']:.3f} "
        f"prefix_cat_ms={metrics['avg_prefix_cat_ms']:.3f} "
        f"attention_bias_ms={metrics['avg_attention_bias_ms']:.3f} "
        f"size_bias_ms={metrics['avg_size_bias_ms']:.3f} "
        f"merge_block_forward_ms={metrics['avg_merge_block_forward_ms']:.3f} "
        f"sparse_state_commit_ms={metrics['avg_sparse_state_commit_ms']:.3f} "
        f"sparse_decoder_input_ms={metrics['avg_sparse_decoder_input_ms']:.3f} "
        f"restore_index_prep_ms={metrics['avg_restore_index_prep_ms']:.3f} "
        f"dark_prior_ms={metrics['avg_dark_prior_ms']:.3f} "
        f"post_norm1_ms={metrics['avg_post_norm1_ms']:.3f} "
        f"post_qkv_ms={metrics['avg_post_qkv_ms']:.3f} "
        f"post_rope_apply_ms={metrics['avg_post_rope_apply_ms']:.3f} "
        f"post_attn_ms={metrics['avg_post_attn_ms']:.3f} "
        f"post_attn_proj_ms={metrics['avg_post_attn_proj_ms']:.3f} "
        f"post_norm2_ms={metrics['avg_post_norm2_ms']:.3f} "
        f"post_mlp_fc1_ms={metrics['avg_post_mlp_fc1_ms']:.3f} "
        f"post_mlp_act_ms={metrics['avg_post_mlp_act_ms']:.3f} "
        f"post_mlp_fc2_ms={metrics['avg_post_mlp_fc2_ms']:.3f} "
        f"post_residual_ms={metrics['avg_post_residual_ms']:.3f} "
        f"post_layout_ms={metrics['avg_post_layout_ms']:.3f}"
    )
    print(f"[speed_real] metrics saved to {metrics_path}")
    print("[speed_real] " + " ".join(f"{key}={value}" for key, value in metrics.items()))
    return metrics
